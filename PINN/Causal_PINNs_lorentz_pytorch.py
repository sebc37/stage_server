import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from scipy.integrate import odeint as scipy_odeint
from tqdm import trange
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = "/Odyssey/private/s26calme/code_stage/PINN/"

# ── MLP (Vanilla) ─────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        net = []
        for d_in, d_out in zip(layers[:-2], layers[1:-1]):
            linear = nn.Linear(d_in, d_out)
            # Glorot / Xavier uniform initialisation
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            net.append(linear)
            net.append(nn.Tanh())
        # Last layer – no activation
        last = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(last.weight)
        nn.init.zeros_(last.bias)
        net.append(last)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# ── Modified MLP (with multiplicative gating) ─────────────────────────────────
class ModifiedMLP(nn.Module):
    """Encoder-gated MLP (Wang et al. 2022 style)."""
    def __init__(self, layers):
        super().__init__()
        d_in, d_h = layers[0], layers[1]

        self.U_enc = nn.Linear(d_in, d_h)
        self.V_enc = nn.Linear(d_in, d_h)
        nn.init.xavier_uniform_(self.U_enc.weight); nn.init.zeros_(self.U_enc.bias)
        nn.init.xavier_uniform_(self.V_enc.weight); nn.init.zeros_(self.V_enc.bias)

        self.hidden = nn.ModuleList()
        for d_i, d_o in zip(layers[1:-2], layers[2:-1]):
            lin = nn.Linear(d_i, d_o)
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
            self.hidden.append(lin)

        self.out = nn.Linear(layers[-2], layers[-1])
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        U = torch.tanh(self.U_enc(x))
        V = torch.tanh(self.V_enc(x))
        h = x
        for lin in self.hidden:
            h = torch.tanh(lin(h))
            h = h * U + (1.0 - h) * V
        return self.out(h)


# ── PINN ──────────────────────────────────────────────────────────────────────
class PINN:
    # Lorenz parameters
    rho   = 28.0
    sigma = 10.0
    beta  = 8.0 / 3.0

    def __init__(self, layers, states0, t0, t1, tol):
        self.states0 = torch.tensor(states0, dtype=torch.float64, device=device)
        self.t0  = t0
        self.t1  = t1
        self.tol = tol

        # Collocation points
        n_t  = 300
        eps  = 0.1 * t1
        t_np = np.linspace(t0, t1 + eps, n_t)
        self.t = torch.tensor(t_np, dtype=torch.float64, device=device)   # (n_t,)

        # Causal weight matrix  M[i,j] = 1  if j < i  (strictly lower triangular → upper triu transposed)
        M = np.triu(np.ones((n_t, n_t)), k=1).T
        self.M = torch.tensor(M, dtype=torch.float64, device=device)      # (n_t, n_t)

        # Network
        self.net = MLP(layers).double().to(device)

        # Adam with exponential LR decay (matches JAX schedule)
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9 ** (1 / 5000))

        # Logs
        self.itercount    = itertools.count()
        self.loss_log     = []
        self.loss_ics_log = []
        self.loss_res_log = []

    # ── Neural network output (scalar interface) ──────────────────────────────
    def neural_net(self, t_scalar):
        """
        t_scalar : 0-d or 1-d scalar tensor.
        Returns x, y, z as scalar tensors.
        """
        t_in = t_scalar.reshape(1, 1)          # (1, 1)
        out  = self.net(t_in) * t_scalar        # (1, 3)
        xyz  = out[0] + self.states0            # (3,)
        return xyz[0], xyz[1], xyz[2]

    # ── Vectorised forward pass over all collocation points ──────────────────
    def _forward_batch(self, t_batch):
        """
        t_batch : (n_t,) tensor with requires_grad=True
        Returns xyz : (n_t, 3) tensor, each row = [x(t), y(t), z(t)]
        """
        t_in = t_batch.unsqueeze(1)                      # (n_t, 1)
        out  = self.net(t_in) * t_batch.unsqueeze(1)     # (n_t, 3)  – hard IC
        return out + self.states0.unsqueeze(0)            # broadcast IC offset

    # ── Residuals (fully vectorised with autograd) ────────────────────────────
    def residuals_and_weights(self):
        # We need gradients w.r.t. t even when called inside torch.no_grad(),
        # so we always work with a fresh leaf that has requires_grad=True.
        t = self.t.detach().requires_grad_(True)          # (n_t,)

        # Enable grad locally even if the caller disabled it
        with torch.enable_grad():
            xyz = self._forward_batch(t)                  # (n_t, 3)
            x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

            # ∂/∂t via a single backward-compatible pass using grad_outputs
            ones = torch.ones_like(x)
            x_t = torch.autograd.grad(x, t, grad_outputs=ones, create_graph=True)[0]
            y_t = torch.autograd.grad(y, t, grad_outputs=ones, create_graph=True)[0]
            z_t = torch.autograd.grad(z, t, grad_outputs=ones, create_graph=True)[0]

        res1 = x_t - self.sigma * (y - x)
        res2 = y_t - x * (self.rho - z) + y
        res3 = z_t - x * y + self.beta * z

        # Causal weights (stop gradient)
        W = torch.exp(
            -self.tol * self.M @ (res1.detach()**2 + res2.detach()**2 + res3.detach()**2)
        )
        return res1, res2, res3, W

    # ── Initial-condition loss ────────────────────────────────────────────────
    def loss_ics(self):
        t0_t = torch.tensor([[self.t0]], dtype=torch.float64, device=device)
        out  = self.net(t0_t) * t0_t                      # (1, 3)
        pred = out[0] + self.states0
        return torch.sum((self.states0 - pred) ** 2)

    # ── Physics residual loss ─────────────────────────────────────────────────
    def loss_res(self):
        r1, r2, r3, W = self.residuals_and_weights()
        return torch.mean(W * (r1**2 + r2**2 + r3**2))

    # ── Total loss ────────────────────────────────────────────────────────────
    def loss(self):
        return self.loss_res()

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self, nIter=10000):
        pbar = trange(nIter)
        for it in pbar:
            self.optimizer.zero_grad()
            loss_val = self.loss()
            loss_val.backward()
            self.optimizer.step()
            self.scheduler.step()

            if it % 1000 == 0:
                # residuals_and_weights uses torch.enable_grad() internally
                # so we just detach at the end – no outer no_grad wrapper needed
                loss_ics_val = self.loss_ics().item()
                r1, r2, r3, W = self.residuals_and_weights()
                loss_res_val  = torch.mean(W * (r1**2 + r2**2 + r3**2)).item()
                w_min         = W.min().item()

                self.loss_log.append(loss_val.item())
                self.loss_ics_log.append(loss_ics_val)
                self.loss_res_log.append(loss_res_val)

                pbar.set_postfix({
                    'Loss':     f'{loss_val.item():.3e}',
                    'loss_ics': f'{loss_ics_val:.3e}',
                    'loss_res': f'{loss_res_val:.3e}',
                    'W_min':    f'{w_min:.4f}',
                })

                if w_min > 0.99:
                    break

    # ── Prediction ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict_u(self, t_array):
        """
        t_array : 1-D numpy array or torch tensor of time points.
        Returns numpy arrays x_pred, y_pred, z_pred.
        """
        if not isinstance(t_array, torch.Tensor):
            t_array = torch.tensor(t_array, dtype=torch.float64, device=device)
        xyz = self._forward_batch(t_array)          # (n_t, 3) – no grad needed
        return (xyz[:, 0].cpu().numpy(),
                xyz[:, 1].cpu().numpy(),
                xyz[:, 2].cpu().numpy())


# ── Reference ODE (Lorenz) ────────────────────────────────────────────────────
rho, sigma, beta = 28.0, 10.0, 8.0 / 3.0

def f(state, t):
    x, y, z = state
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    T      = 30
    t0     = 0.0
    t1     = 0.5
    layers = [1, 512, 512, 512, 3]

    tol_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1]

    x_pred_list, y_pred_list, z_pred_list = [], [], []
    params_list, losses_list = [], []

    state0 = np.array([1.0, 1.0, 1.0])
    t_local = np.arange(t0, t1, 0.01)

    for k in range(int(T / t1)):
        print(f'\n=== Final Time: {(k+1) * t1:.2f} ===')
        model = PINN(layers, state0, t0, t1, tol=tol_list[0])

        for tol in tol_list:
            model.tol = tol
            print(f'  tol: {tol}')
            model.train(nIter=300000)

        # Predictions on the local time window
        x_pred, y_pred, z_pred = model.predict_u(t_local)

        # New initial condition = network output at t1
        t1_t = torch.tensor(t1, dtype=torch.float64, device=device)
        with torch.no_grad():
            x0, y0, z0 = model.neural_net(t1_t)
        state0 = np.array([x0.item(), y0.item(), z0.item()])

        x_pred_list.append(x_pred)
        y_pred_list.append(y_pred)
        z_pred_list.append(z_pred)
        losses_list.append([model.loss_ics_log, model.loss_res_log])

        # Save predictions
        np.save(PATH + 'x_pred_list.npy', np.array(x_pred_list, dtype=object))
        np.save(PATH + 'y_pred_list.npy', np.array(y_pred_list, dtype=object))
        np.save(PATH + 'z_pred_list.npy', np.array(z_pred_list, dtype=object))
        np.save(PATH + 'losses_list.npy',  np.array(losses_list, dtype=object))

        # Error vs reference ODE
        t_star = np.arange(t0, (k + 1) * t1, 0.01)
        states = scipy_odeint(f, [1.0, 1.0, 1.0], t_star)

        x_preds = np.concatenate(x_pred_list)
        y_preds = np.concatenate(y_pred_list)
        z_preds = np.concatenate(z_pred_list)

        err_x = np.linalg.norm(x_preds - states[:, 0]) / np.linalg.norm(states[:, 0])
        err_y = np.linalg.norm(y_preds - states[:, 1]) / np.linalg.norm(states[:, 1])
        err_z = np.linalg.norm(z_preds - states[:, 2]) / np.linalg.norm(states[:, 2])
        print(f'Relative l2 error  x: {err_x:.3e}')
        print(f'Relative l2 error  y: {err_y:.3e}')
        print(f'Relative l2 error  z: {err_z:.3e}')

    # ── Final 3-D plot ────────────────────────────────────────────────────────
    t_full = np.arange(0.0, T, 0.01)
    ref    = scipy_odeint(f, [1.0, 1.0, 1.0], t_full)

    fig = plt.figure(figsize=(14, 5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(ref[:, 0], ref[:, 1], ref[:, 2], 'b', lw=0.5, label='Reference')
    ax1.set_title('Reference Lorenz attractor')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    xp = np.concatenate(x_pred_list)
    yp = np.concatenate(y_pred_list)
    zp = np.concatenate(z_pred_list)
    ax2.plot(xp, yp, zp, 'r', lw=0.5, label='PINN')
    ax2.set_title('PINN prediction')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PATH + 'lorenz_pinn_pytorch.png', dpi=150)
    plt.show()
