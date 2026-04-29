import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import os

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
PATH = "/Odyssey/private/s26calme/code_stage/PINN/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
#  Early-stopping callback equivalent
# ─────────────────────────────────────────────
class EarlyStopping:
    """Stop training when loss has not improved for `patience` epochs."""

    def __init__(self, patience: int = 1000):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, loss: float, model: nn.Module) -> bool:
        """Return True when training should stop."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
#  Fourier Feature Layer
# ─────────────────────────────────────────────
class FourierLayer(nn.Module):
    """Random Fourier Features (non-trainable projection)."""

    def __init__(self, input_dim: int, output_dim: int, scale: float = 10.0):
        super().__init__()
        self.scale = scale
        # Fixed (non-trainable) random matrix B
        B = torch.randn(input_dim, output_dim) * 1.0
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = x @ self.B * self.scale          # (batch, output_dim)
        return torch.cat([torch.sin(projection),
                          torch.cos(projection)], dim=-1)  # (batch, 2*output_dim)


# ─────────────────────────────────────────────
#  PINN model
# ─────────────────────────────────────────────
class ODE2nd(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 3,
        fourier_dim: int = 100,
        scale: float = 10.0,
        hidden: int = 500,
        activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.fourier = FourierLayer(input_dim, fourier_dim, scale)
        fourier_out = 2 * fourier_dim

        self.net = nn.Sequential(
            nn.Linear(fourier_out, hidden), activation,
            nn.Linear(hidden, hidden),     activation,
            nn.Linear(hidden, hidden),     activation,
            nn.Linear(hidden, output_dim),
        )

        # Xavier / Glorot uniform initialisation (matches TF default)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.fourier(x))


# ─────────────────────────────────────────────
#  Lorenz ODE reference solution (Runge-Kutta 4)
# ─────────────────────────────────────────────
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])


def runge_kutta(f, y0, t_0, t_f, h, *args):
    t_values = np.arange(t_0, t_f + h, h)
    print(f"RK4 steps: {t_values.shape[0]}")
    n = len(t_values)
    y_values = np.zeros((n, len(y0)))
    y_values[0] = y0
    for i in range(1, n):
        k1 = h * f(t_values[i - 1],             y_values[i - 1],             *args)
        k2 = h * f(t_values[i - 1] + 0.5 * h,  y_values[i - 1] + 0.5 * k1, *args)
        k3 = h * f(t_values[i - 1] + 0.5 * h,  y_values[i - 1] + 0.5 * k2, *args)
        k4 = h * f(t_values[i - 1] + h,         y_values[i - 1] + k3,       *args)
        y_values[i] = y_values[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return t_values, y_values


# ─────────────────────────────────────────────
#  Loss function (Lorenz residuals + IC penalty)
# ─────────────────────────────────────────────
def compute_loss(
    model: nn.Module,
    x: torch.Tensor,          # (N, 1) collocation points, requires_grad=True
    x0: torch.Tensor,         # (1, 1) initial point
    y0_true: torch.Tensor,    # (1, 3) initial condition
    n_train: int,
    mse: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # ── ODE residual ──────────────────────────────────────────────
    x = x.requires_grad_(True)
    y = model(x)                    # (N, 3)

    # Gradient of each output w.r.t. x  →  shape (N, 3)
    dy_dx = torch.zeros_like(y)
    for j in range(3):
        grad = torch.autograd.grad(
            y[:, j].sum(), x,
            create_graph=True,
            retain_graph=True,
        )[0]                        # (N, 1)
        dy_dx[:, j] = grad[:, 0]

    a = torch.tensor(10.0,      device=device)
    b = torch.tensor(28.0,      device=device)
    c = torch.tensor(8.0 / 3.0, device=device)

    rhs0 = a * (y[:, 1] - y[:, 0])
    rhs1 = y[:, 0] * (b - y[:, 2]) - y[:, 1]
    rhs2 = y[:, 0] * y[:, 1] - c * y[:, 2]

    loss_ode = (
        mse(dy_dx[:, 0], rhs0) +
        mse(dy_dx[:, 1], rhs1) +
        mse(dy_dx[:, 2], rhs2)
    ) / n_train

    # ── Initial-condition loss ─────────────────────────────────────
    y0_pred = model(x0)             # (1, 3)
    loss_bc = mse(y0_pred, y0_true) * 10.0

    return loss_ode + loss_bc, loss_ode, loss_bc


# ─────────────────────────────────────────────
#  Training loop for one interval
# ─────────────────────────────────────────────
def train_interval(
    model: nn.Module,
    optimizer: optim.Optimizer,
    x_train: torch.Tensor,
    x0: torch.Tensor,
    y0_true: torch.Tensor,
    n_train: int,
    epochs: int,
    batch_size: int,
    checkpoint_path: str,
    target_loss: float = 5e-6,
    patience: int = 1000,
) -> dict:

    mse = nn.MSELoss()
    early_stop = EarlyStopping(patience=patience)
    history = {"lossreal": [], "lossODE": [], "lossBC": []}
    best_loss = float("inf")

    dataset = TensorDataset(x_train)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        epoch_loss = epoch_ode = epoch_bc = 0.0
        n_batches = 0

        for (x_batch,) in loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            loss, l_ode, l_bc = compute_loss(
                model, x_batch, x0, y0_true, n_train, mse
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_ode  += l_ode.item()
            epoch_bc   += l_bc.item()
            n_batches  += 1

        avg_loss = epoch_loss / n_batches
        avg_ode  = epoch_ode  / n_batches
        avg_bc   = epoch_bc   / n_batches

        history["lossreal"].append(avg_loss)
        history["lossODE"].append(avg_ode)
        history["lossBC"].append(avg_bc)

        # Save best weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)

        # Target-loss stop
        if avg_loss <= target_loss:
            break

        # Early stopping
        if early_stop.step(avg_loss, model):
            break

    # Restore best weights for this interval
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return history


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────
def main():
    # ── Read input file ───────────────────────────────────────────
    ruta_ini = PATH + "input_chaos.txt"
    with open(ruta_ini, "r") as f:
        for line in f:
            vals = line.split()
            N_train     = int(vals[0])
            N_intervalos = int(vals[1])
            epochs      = int(vals[2])
            xmax        = float(vals[3])

    lr       = 0.001
    salto_x  = xmax / N_intervalos

    # ── Lorenz parameters ─────────────────────────────────────────
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    y0_np = np.array([1.0, 1.0, 1.0])
    t_0, t_f, h = 0.0, xmax, 0.0001

    # ── RK4 reference trajectory ──────────────────────────────────
    t_values, y_values = runge_kutta(lorenz, y0_np, t_0, t_f, h, sigma, rho, beta)

    # ── Hyper-parameters ──────────────────────────────────────────
    fourier_features_dim = 100
    scale                = 10.0
    batch_size           = 25
    input_neurons        = 1
    output_neurons       = 3

    print(f"N_train={N_train}")
    print(f"N_intervalos={N_intervalos}")
    print(f"x_max={xmax}")
    print(f"epochs={epochs}")

    os.makedirs("p-caos", exist_ok=True)

    auxx, aux2 = [], []
    x0   = 0.0
    xmed = x0
    y0   = y0_np.copy()

    last_history = None

    for chinch in range(N_intervalos):
        checkpoint_path = (
            f"p-caos/pesos_inter={N_intervalos}_epochs={epochs}_x={xmax}.pt"
        )

        # Build model & optimiser fresh for each interval
        model = ODE2nd(
            input_dim=input_neurons,
            output_dim=output_neurons,
            fourier_dim=fourier_features_dim,
            scale=scale,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        xmed += salto_x
        x_train_np = np.linspace(x0, xmed, N_train).reshape(-1, 1).astype(np.float32)
        x_train    = torch.tensor(x_train_np, device=device)

        x0_t    = torch.tensor([[x0]], dtype=torch.float32, device=device)
        y0_true = torch.tensor([y0],  dtype=torch.float32, device=device)  # (1,3)

        last_history = train_interval(
            model, optimizer,
            x_train, x0_t, y0_true,
            N_train, epochs, batch_size,
            checkpoint_path,
        )

        # Predict over this interval
        model.eval()
        with torch.no_grad():
            xy_pred = model(x_train).cpu().numpy()  # (N, 3)

        last_pred = xy_pred[N_train - 1]   # endpoint state
        last_x    = x_train_np[N_train - 1]

        auxx.append(last_pred)
        aux2.append(last_x)

        # Append endpoint to aux file
        ruta_aux = f"p-caos/aux_Ninter={N_intervalos}_epochs={epochs}_x={xmax}.txt"
        with open(ruta_aux, "a") as f:
            f.write(
                f"{last_x[0]:.8f}\t{last_pred[0]:.8f}\t"
                f"{last_pred[1]:.8f}\t{last_pred[2]:.8f}\n"
            )

        # Advance interval
        x0 += salto_x
        y0 = last_pred.copy()

    # ── Save "chinchetas" (pin-point trajectory) ──────────────────
    chinchetas     = np.array(auxx)          # (N_intervalos, 3)
    chincheta_aux  = [x[0] for x in aux2]   # list of floats

    ruta_chinchetas = (
        f"p-caos/chinchetas_Ninter={N_intervalos}_epochs={epochs}_x={xmax}.txt"
    )
    with open(ruta_chinchetas, "w") as f:
        for t, cx, cy, cz in zip(
            chincheta_aux,
            chinchetas[:, 0],
            chinchetas[:, 1],
            chinchetas[:, 2],
        ):
            f.write(f"{t:.8f}\t{cx:.8f}\t{cy:.8f}\t{cz:.8f}\n")

    # ── Save loss history of the last interval ────────────────────
    ruta_loss = f"p-caos/Ninter={N_intervalos}_epochs={epochs}_x={xmax}.txt"
    with open(ruta_loss, "w") as f:
        for lr_, lo_, lb_ in zip(
            last_history["lossreal"],
            last_history["lossODE"],
            last_history["lossBC"],
        ):
            f.write(f"{lr_:.10f}\t{lo_:.10f}\t{lb_:.10f}\n")

    print("Done. Output files written to p-caos/")


if __name__ == "__main__":
    main()
