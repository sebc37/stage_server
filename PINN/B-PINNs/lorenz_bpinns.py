"""
Lorenz System - Bayesian Physics-Informed Neural Networks (BPINNs)
===================================================================
Adapted from 1dnonlinear.py. Infers the Lorenz ODE trajectory
  dx/dt = sigma*(y - x)
  dy/dt = x*(rho - z) - y
  dz/dt = x*y - beta*z
using a single multi-output network u(t) -> [x(t), y(t), z(t)].

The loss has two terms:
  1. Data likelihood  : sparse noisy observations of (x,y,z)
  2. ODE residual     : collocation points where the Lorenz equations
                        must be satisfied (PDE-style residual)
"""

import torch
import torch.nn as nn
import hamiltorch
import matplotlib.pyplot as plt
import numpy as np
import util  # your local util.py

PATH = "/Odyssey/private/s26calme/code_stage/PINN/B-PINNs/"

# ─── Device ──────────────────────────────────────────────────────────────────
print(f'Is CUDA available?: {torch.cuda.is_available()}')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Lorenz parameters ───────────────────────────────────────────────────────
sigma = 10.0
rho   = 28.0
beta  = 8.0 / 3.0

# ─── Time domain ─────────────────────────────────────────────────────────────
t0, t1 = 0.0, 10.0        # short window so HMC stays tractable
N_tr_u = 100              # observed data points
N_tr_f = 128              # collocation (ODE residual) points
N_val  = 10000            # validation grid

# ─── Ground-truth trajectory via RK4 ─────────────────────────────────────────
def lorenz_rhs(state, sigma=sigma, rho=rho, beta=beta):
    x, y, z = state
    return [sigma*(y - x), x*(rho - z) - y, x*y - beta*z]

def rk4(state0, ts):
    states = [state0]
    for i in range(len(ts) - 1):
        dt = ts[i+1] - ts[i]
        s  = states[-1]
        k1 = lorenz_rhs(s)
        k2 = lorenz_rhs([s[j] + 0.5*dt*k1[j] for j in range(3)])
        k3 = lorenz_rhs([s[j] + 0.5*dt*k2[j] for j in range(3)])
        k4 = lorenz_rhs([s[j] +     dt*k3[j] for j in range(3)])
        states.append([s[j] + (dt/6)*(k1[j]+2*k2[j]+2*k3[j]+k4[j]) for j in range(3)])
    return np.array(states)   # (T, 3)

ts_dense = np.linspace(t0, t1, 10000)
xyz0     = [8.0, 0.0, 30.0]
traj     = rk4(xyz0, ts_dense)   # (2000, 3)
#traj_bruit = traj + np.random.randn(*traj.shape) * 0.5   # add small noise to make it more realistic

def true_xyz(t_tensor):
    """Interpolate ground-truth trajectory at query times (numpy-backed)."""
    t_np = t_tensor.detach().numpy().flatten()
    xyz  = np.stack([np.interp(t_np, ts_dense, traj[:, i]) for i in range(3)], axis=1) #traj
    return torch.tensor(xyz, dtype=torch.float32)


# ─── Hyperparameters ─────────────────────────────────────────────────────────
hamiltorch.set_random_seed(123)
prior_std   = 1.0
like_std    = 1.0
step_size   = 0.0001       # small: Lorenz gradients are O(10-100) near the attractor
burn        = 200
num_samples = 5000
L           = 100
layer_sizes = [1, 64, 64, 64, 64, 3]   # t -> (x, y, z)
activation  = torch.tanh
pde         = True
pinns       = True          # warm-start with MAP/PINNs, then switch to HMC below
epochs      = 2*80000
tau_priors  = 1.0 / prior_std**2
# tau_likes must be a list [obs_precision, ode_precision] when pde=True.
# Lorenz RHS values are O(10-100), so a large ode precision explodes gradients.
tau_obs   = 1 / like_std**2   # 100 -- tight on observations
tau_ode   = 1.                  # loose on physics to keep gradients finite
tau_likes = [tau_obs, tau_ode]

# ─── Build datasets ──────────────────────────────────────────────────────────
t_obs = torch.linspace(t0, t1, N_tr_u).view(-1, 1)
xyz_obs = true_xyz(t_obs) + torch.randn(N_tr_u, 3) * like_std

t_col = torch.linspace(t0, t1, N_tr_f).view(-1, 1)

t_val = torch.linspace(t0, t1, N_val).view(-1, 1)
xyz_val = true_xyz(t_val)

data = {
    'x_u': t_obs,     # input times for observations
    'y_u': xyz_obs,   # observed (x,y,z), shape (N_tr_u, 3)
    'x_f': t_col,     # collocation times
}
data_val = {
    'x_u': t_val,
    'y_u': xyz_val,
    'x_f': t_val,
}

for k in data:
    data[k] = data[k].to(device)
for k in data_val:
    data_val[k] = data_val[k].to(device)

# ─── Neural network ──────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, layer_sizes, activation=torch.tanh):
        super().__init__()
        self.activation = activation
        self.l1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.l2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.l3 = nn.Linear(layer_sizes[2], layer_sizes[3])
        self.l4 = nn.Linear(layer_sizes[3], layer_sizes[4])
        self.l5 = nn.Linear(layer_sizes[4], layer_sizes[5])
        
    def forward(self, x):
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        x = self.activation(self.l3(x))
        x = self.activation(self.l4(x))
        return self.l5(x)

net = Net(layer_sizes, activation).to(device)
nets = [net]

# ─── Model loss (data + ODE residuals) ───────────────────────────────────────
def model_loss(data, fmodel, params_unflattened, tau_likes, gradients, params_single=None):
    """
    fmodel[0] : t -> (x, y, z)   shape (N, 3)

    Returns
    -------
    ll     : scalar log-likelihood
    output : [pred_u, pred_f]  (for prediction)
    """
    # --- observation likelihood ---
    t_u   = data['x_u']
    xyz_u = data['y_u']
    pred_u = fmodel[0](t_u, params=params_unflattened[0])       # (N_tr_u, 3)
    ll = -0.5 * tau_likes[0] * ((pred_u - xyz_u) ** 2).sum()

    # --- ODE residual likelihood ---
    t_f = data['x_f'].detach().requires_grad_()
    uvw = fmodel[0](t_f, params=params_unflattened[0])           # (N_tr_f, 3)
    u = uvw[:, 0:1]
    v = uvw[:, 1:2]
    w = uvw[:, 2:3]

    u_t = gradients(u, t_f)[0]   # du/dt, shape (N_tr_f, 1)
    v_t = gradients(v, t_f)[0]
    w_t = gradients(w, t_f)[0]

    # Lorenz residuals
    res_x = u_t - sigma * (v - u)
    res_y = v_t - (u * (rho - w) - v)
    res_z = w_t - (u * v - beta * w)

    ode_sq = res_x**2 + res_y**2 + res_z**2
    # Guard: if any value is NaN/Inf, return a large but finite penalty
    if torch.isnan(ode_sq).any() or torch.isinf(ode_sq).any():
        ode_sq = torch.zeros_like(ode_sq)

    ll = ll - 0.5 * tau_likes[1] * ode_sq.sum()

    pred_f = torch.cat([res_x, res_y, res_z], dim=1)   # (N_tr_f, 3)
    output = [pred_u, pred_f]

    return ll, output

# ─── Sampling ────────────────────────────────────────────────────────────────
# Phase 1: PINNs warm-start to find a good MAP initialisation for HMC
print('Phase 1: PINNs warm-start (MAP optimisation)...')
params_map = util.sample_model_bpinns(
    nets, data, model_loss=model_loss,
    num_samples=num_samples, num_steps_per_sample=L,
    step_size=1e-3, burn=burn,          # Adam lr, not leapfrog step_size
    tau_priors=tau_priors, tau_likes=tau_likes,
    device=device, pde=pde, pinns=True, epochs=epochs,
)

# Phase 2: HMC from MAP initialisation
print('Phase 2: HMC sampling from MAP init...')
params_hmc = util.sample_model_bpinns(
    nets, data, model_loss=model_loss,
    num_samples=num_samples, num_steps_per_sample=L,
    step_size=step_size, burn=burn,
    tau_priors=tau_priors, tau_likes=tau_likes,
    device=device, pde=pde, pinns=False, epochs=epochs,
    params_init_val=params_map[0],      # warm-start HMC from MAP
)

pred_list, log_prob_list = util.predict_model_bpinns(
    nets, params_hmc, data_val, model_loss=model_loss,
    tau_priors=tau_priors, tau_likes=tau_likes, pde=pde,
)

print('\nExpected validation log probability: {:.3f}'.format(
    torch.stack(log_prob_list).mean()))

# pred_list[0]: (S, N_val, 3)  — trajectory predictions
pred_xyz = pred_list[0].cpu().numpy()   # (S, N_val, 3)

# ─── Plots ───────────────────────────────────────────────────────────────────
t_np   = t_val.cpu().numpy().flatten()
xyz_np = xyz_val.cpu().numpy()   # (N_val, 3)
labels = ['x(t)', 'y(t)', 'z(t)']
colors = ['steelblue', 'darkorange', 'seagreen']

fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
    mean = pred_xyz.mean(0)[:, i]
    std  = pred_xyz.std(0)[:, i]
    ax.plot(t_np, xyz_np[:, i], 'r-', lw=2, label='Vérité terrain')
    ax.plot(t_np, mean, color=col, lw=1.5, label='Moyenne BPINN')
    ax.fill_between(t_np, mean - 2*std, mean + 2*std,
                    color=col, alpha=0.25, label='±2 std')
    ax.scatter(data['x_u'].cpu().numpy(),
               data['y_u'].cpu().numpy()[:, i],
               c='k', s=20, zorder=5, label='Obs. data')

    # ── Points de collocation (résidus ODE) ──────────────────────────
    t_col_np = data['x_f'].cpu().numpy().flatten()
    # Évaluer u(t_col) pour récupérer la valeur de la composante i
    with torch.no_grad():
        pred_col = net(data['x_f']).cpu().numpy()[:, i]
    ax.scatter(t_col_np, pred_col,
               marker='|', c='purple', s=40, linewidths=0.8,
               alpha=0.5, zorder=4, label='points de collocation')
    # ─────────────────────────────────────────────────────────────────

    ax.set_ylabel(lbl, fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim([t0, t1])

# fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
# for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
#     mean = pred_xyz.mean(0)[:, i]
#     std  = pred_xyz.std(0)[:, i]
#     ax.plot(t_np, xyz_np[:, i], 'r-', lw=2, label='Ground truth')
#     ax.plot(t_np, mean, color=col, lw=1.5, label='BPINN mean')
#     ax.fill_between(t_np, mean - 2*std, mean + 2*std,
#                     color=col, alpha=0.25, label='±2 std')
#     ax.scatter(data['x_u'].cpu().numpy(),
#                data['y_u'].cpu().numpy()[:, i],
#                c='k', s=20, zorder=5, label='Obs. data')
#     ax.set_ylabel(lbl, fontsize=12)
#     ax.legend(fontsize=9)
#     ax.set_xlim([t0, t1])

axes[-1].set_xlabel('t', fontsize=12)
plt.suptitle('Système de Lorenz — BPINNs (HMC)', fontsize=14)
plt.tight_layout()
plt.savefig(PATH + 'lorenz_bpinns_timeseries.png', dpi=150)
plt.show()

# --- 3-D phase portrait ---
fig3d = plt.figure(figsize=(8, 6))
ax3d  = fig3d.add_subplot(111, projection='3d')
ax3d.plot(xyz_np[:, 0], xyz_np[:, 1], xyz_np[:, 2], 'r-', lw=1, label='Vérité terrain')
ax3d.plot(pred_xyz.mean(0)[:, 0],
          pred_xyz.mean(0)[:, 1],
          pred_xyz.mean(0)[:, 2],
          'b--', lw=1, label='Moyenne BPINN')
ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
ax3d.set_title('Attracteur de Lorenz — portrait de phase')
ax3d.legend()
plt.tight_layout()
plt.savefig(PATH + 'lorenz_bpinns_phase.png', dpi=150)
plt.show()

# ── RMSE par composante en fonction du temps ──────────────────────────────────
rmse = np.sqrt(((pred_xyz.mean(0) - xyz_np) ** 2))   # (N_val, 3)  — erreur absolue par pas


fig_rmse, ax_rmse = plt.subplots(figsize=(9, 3))
for i, (lbl, col) in enumerate(zip(labels, colors)):
    ax_rmse.plot(t_np, rmse[:, i], color=col, lw=1.5, label=lbl)

ax_rmse.set_xlabel('t', fontsize=12)
ax_rmse.set_ylabel('Erreur absolue (|ŷ − y|)', fontsize=11)
ax_rmse.set_title('Erreur point-à-point — BPINNs vs Vérité terrain', fontsize=13)
ax_rmse.legend(fontsize=10)
ax_rmse.set_xlim([t0, t1])
ax_rmse.set_yscale('log')          # log-scale utile car l'erreur diverge sur l'attracteur
plt.tight_layout()
plt.savefig(PATH + 'lorenz_bpinns_rmse.png', dpi=150)
plt.show()

# ── RMSE global (scalaire) par composante ─────────────────────────────────────
for i, lbl in enumerate(labels):
    val = np.sqrt(((pred_xyz.mean(0)[:, i] - xyz_np[:, i]) ** 2).mean())
    print(f'RMSE {lbl} : {val:.4f}')