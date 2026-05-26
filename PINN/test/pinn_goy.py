import os
import numpy as np
from custom_sampler import *
from architecture import *
from parser import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim
import tqdm
import matplotlib.pyplot as plt
import argparse


# ─────────────────────────────────────────────────────────────────────────────
#  Classe d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
class Train_PINN():
    """
    Entraîne le PINN sur :
      • des observations tirées uniformément sur (k,t,u)  → fixes
      • des points de résidus tirés aléatoirement sur (k,t) → fixes
    Les deux ensembles sont générés UNE SEULE FOIS avant l'entraînement.
    """

    def __init__(self, learning_rate, nbr_iteration, w_obs, w_phy,
                 physic=True, normalize_phy=True):
        self.learning_rate  = learning_rate
        self.nbr_iteration  = nbr_iteration
        self.w_obs          = w_obs
        self.w_phy          = w_phy
        self.optimizer      = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.physic         = physic
        self.normalize_phy  = normalize_phy

    # ── boucle d'entraînement ─────────────────────────────────────────────
    def train(self, obs_loader, res_loader):
        """
        Args:
            obs_loader : DataLoader issu de observation_dataset  (k, t_norm, u)
            res_loader : DataLoader issu de residual_dataset     (k, t_norm)
        Returns:
            (loss_total, loss_obs, loss_phy)  — tableaux numpy (nbr_iteration,)
        """
        loss_tracker     = np.zeros(self.nbr_iteration)
        loss_obs_tracker = np.zeros(self.nbr_iteration)
        loss_phy_tracker = np.zeros(self.nbr_iteration)

        torch.autograd.set_detect_anomaly(True)

        for iteration in tqdm.tqdm(range(self.nbr_iteration)):
            self.optimizer.zero_grad()

            # ── 1. Loss d'observation ─────────────────────────────────────
            loss_obs = torch.tensor(0.0, device=device)
            for k_batch, t_batch, u_batch in obs_loader:
                k_b = k_batch.to(device)
                t_b = t_batch.to(device)
                u_b = u_batch.to(device)

                inp = torch.stack([k_b, t_b], dim=1)          # (B, 2)
                u_pred = model(inp).squeeze(-1)                # (B,)
                loss_obs = loss_obs + torch.mean((u_pred - u_b) ** 2)

            # ── 2. Loss physique (résidus GOY) ────────────────────────────
            loss_phy = torch.tensor(0.0, device=device)
            if self.physic:
                for k_batch, t_batch in res_loader:
                    k_b = k_batch.to(device)
                    t_b = t_batch.to(device)

                    inp = torch.stack([k_b, t_b], dim=1)
                    inp.requires_grad_(True)

                    u_pred   = model(inp).squeeze(-1)          # (B,)
                    # dérivée en temps
                    grad_out = torch.autograd.grad(
                        u_pred, inp,
                        grad_outputs=torch.ones_like(u_pred),
                        create_graph=True
                    )[0]                                       # (B, 2)
                    du_dt = grad_out[:, 1]                     # (B,)

                    # ── résidu GOY shell par shell ─────────────────────────
                    residual = torch.zeros_like(u_pred)
                    for i in range(len(k_b)):
                        ki_int = int(k_b[i].item())
                        if ki_int < k_min_collocation or ki_int >= k_max_collocation:
                            continue

                        def u_shell(shell_idx, t_val):
                            inp_s = torch.tensor(
                                [[float(shell_idx), t_val.item()]],
                                device=device, dtype=torch.float32
                            )
                            return model(inp_s).squeeze()

                        Ki    = K[ki_int - k_min_collocation]
                        u_i   = u_pred[i]
                        u_ip1 = u_shell(ki_int + 1, t_b[i])
                        u_ip2 = u_shell(ki_int + 2, t_b[i])
                        u_im1 = u_shell(ki_int - 1, t_b[i]) if ki_int > 0 else torch.tensor(0.0, device=device)
                        u_im2 = u_shell(ki_int - 2, t_b[i]) if ki_int > 1 else torch.tensor(0.0, device=device)

                        goy = (
                            du_dt[i]
                            - Ki * u_ip1 * u_ip2
                            + Ki * eps / lmb * u_im1 * u_ip1
                            + Ki * (eps - 1) / lmb ** 2 * u_im2 * u_im1
                            + nu * Ki ** 2 * u_i
                        )
                        residual[i] = goy

                    loss_phy = loss_phy + torch.mean(residual ** 2)

            # ── 3. Loss totale ────────────────────────────────────────────
            total_loss = self.w_obs * loss_obs + self.w_phy * loss_phy
            total_loss.backward()
            self.optimizer.step()

            loss_tracker[iteration]     = total_loss.item()
            loss_obs_tracker[iteration] = loss_obs.item()
            loss_phy_tracker[iteration] = loss_phy.item()

            if iteration % 500 == 0:
                print(f"iter {iteration:6d} | "
                      f"total={total_loss.item():.4e} | "
                      f"obs={loss_obs.item():.4e} | "
                      f"phy={loss_phy.item():.4e}")

        return loss_tracker, loss_obs_tracker, loss_phy_tracker


# ─────────────────────────────────────────────────────────────────────────────
#  Fonctions utilitaires conservées
# ─────────────────────────────────────────────────────────────────────────────
def filter_mode(X, mode_min, mode_max, t_min, ratio, seed):
    np.random.seed(seed=seed)
    X_subset = np.copy(X)
    X_subset[:, 0:mode_min] = None
    X_subset[0:t_min, :] = None
    nb_column = np.shape(X_subset)[1]
    nb_line   = np.shape(X_subset)[0]
    X_filtered = np.copy(X_subset)
    X_posx, X_posy, X_value, X_dataset = [], [], [], []

    var_mode  = [np.var(X[:, j])  for j in range(nb_column)]
    std_mode  = [np.std(X[:, j])  for j in range(nb_column)]
    mean_mode = [np.mean(X[:, j]) for j in range(nb_column)]

    for j in range(mode_min, mode_max):
        for i in range(t_min, nb_line):
            if np.random.random() <= ratio:
                X_filtered[i, j] = X_subset[i, j]
                X_posx.append(j)
                X_posy.append(i / nb_line)
                X_value.append(X_subset[i, j])
            else:
                X_filtered[i, j] = None

    pourcentage_filtered = nb_line * ratio / nb_line * 100
    X_dataset.extend([X_posx, X_posy, X_value])
    return X_filtered, X_dataset, mean_mode, var_mode, std_mode, pourcentage_filtered


def reduced_center(X, mean, std):
    for k in range(X.shape[1]):
        X[:, k] = (X[:, k] - mean[k]) / std[k]
    return X


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG & DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("config", help="chemin vers YAML avec config du PINN", type=str)
args = parser.parse_args().config

config = parse_config(args)
print(config)

command = torch.cuda.is_available()
print(f'cuda is available : {command}')

PATH      = config["path_graph"]
path_data = config["path_data"]

data        = np.loadtxt(path_data, dtype=np.float32)
Nmax        = np.shape(data)[0]
debut       = int(0.1 * Nmax)
Data_shell  = data[debut:Nmax, :]
Npts        = np.shape(Data_shell)[0]

k_min_collocation = config["k_min_collocation"]
k_max_collocation = config["k_max_collocation"]
k_bc_min          = config["k_min_boundary"]
k_bc_max          = config["k_max_boundary"]
ratio             = config["ratio"]
nb_couche         = config["PINN"][1]
largeur_couche    = config["PINN"][0]
nbr_iteration     = config["nb_iter"]
physic            = config["physic"]
normalize_phy     = config["normalize_phy"]

# Paramètres du modèle GOY
k0       = 0.125
lmb      = 2.0
eps      = 0.5
nu       = 1.e-7
nb_shell = 22
dt       = 8.9999e-5
f        = 99999.9
time     = 1.0
N_fs     = int(1 / ((f - 0.1) * dt))

k_min = min(k_min_collocation, k_bc_min)
k_max = max(k_bc_max, k_max_collocation)
K     = [k0 * lmb ** i for i in range(k_min, k_max)]

t_min = 0.1 * time
t_max = time

print("Moyenne de l'ensemble des modes", np.mean(Data_shell))
print("std de tous les modes ", np.std(Data_shell))

# ─────────────────────────────────────────────────────────────────────────────
#  MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GOY_PINN(n_input=2, n_output=1, n_hidden=largeur_couche, n_layers=nb_couche)
model.to(device)

# ─────────────────────────────────────────────────────────────────────────────
#  DATASETS FIXES  (tirés UNE SEULE FOIS avant l'entraînement)
# ─────────────────────────────────────────────────────────────────────────────
obs_ratio   = float(ratio) / 100.0 if ratio > 1 else float(ratio)   # supporte % ou fraction
N_residual  = int(0.05 * Npts * (k_max_collocation - k_min_collocation))  # ~5 % de la grille physique

obs_ds  = observation_dataset(
    Data_shell, k_min_collocation, k_max_collocation,
    Npts, ratio=obs_ratio, seed=42
)
res_ds  = residual_dataset(
    k_min_collocation, k_max_collocation,
    Npts, N_residual=N_residual, seed=123
)

print(f"Nb points d'observation  : {len(obs_ds)}")
print(f"Nb points de résidu      : {len(res_ds)}")

obs_loader = DataLoader(obs_ds, batch_size=512, shuffle=True,  drop_last=False)
res_loader = DataLoader(res_ds, batch_size=512, shuffle=False, drop_last=False)

# ─────────────────────────────────────────────────────────────────────────────
#  ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
learning_rate = 0.001
t_trainer = Train_PINN(
    learning_rate, nbr_iteration,
    w_obs=1.0, w_phy=1.0,
    physic=bool(physic), normalize_phy=bool(normalize_phy)
)
Total_loss = t_trainer.train(obs_loader, res_loader)

# ─────────────────────────────────────────────────────────────────────────────
#  INFÉRENCE
# ─────────────────────────────────────────────────────────────────────────────
model.eval()

grid_ds = grid_data(k_min_collocation, k_max_collocation, t_min, t_max, Npts=Npts)
with torch.no_grad():
    U_pred = model(grid_ds.grid.to(device)).cpu().numpy()   # (N_k * Npts, 1)

N_k   = grid_ds.N_k
U_pred = U_pred.reshape(N_k, Npts).T                       # (Npts, N_k)
U_exa  = Data_shell[0:Npts, k_min_collocation * 2: k_max_collocation * 2]

# Coordonnées des points d'observation pour les plots
k_obs_arr = obs_ds.k_obs   # indice shell (float)
t_obs_arr = obs_ds.t_obs   # temps normalisé [0,1] → on remet en indice temporel
t_obs_idx = (t_obs_arr * Npts).astype(int).clip(0, Npts - 1)

# Coordonnées des points de résidu pour les plots
k_res_arr = res_ds.k_res   # indice shell
t_res_arr = res_ds.t_res   # temps normalisé
t_res_idx = (t_res_arr * Npts).astype(int).clip(0, Npts - 1)

# ── Calcul RMSE ──────────────────────────────────────────────────────────────
min_cols = min(U_pred.shape[1], U_exa.shape[1])
rmse = np.sqrt(np.mean((U_pred[:, :min_cols] - U_exa[:, :min_cols]) ** 2))
print("RMSE:", rmse)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOTS D'INFÉRENCE  (avec points d'observation et de résidu)
# ─────────────────────────────────────────────────────────────────────────────
t_axis = np.linspace(0, 1, Npts)

for i in range(min_cols):
    shell_global = k_min_collocation * 2 + i   # indice global du shell

    # Sélection des points d'observation appartenant à ce shell
    mask_obs = (k_obs_arr.astype(int) == shell_global)
    t_obs_shell = t_obs_arr[mask_obs]          # temps normalisé
    u_obs_shell = U_exa[t_obs_idx[mask_obs].clip(0, Npts-1), i]

    # Sélection des points de résidu appartenant à ce shell
    mask_res = (k_res_arr.astype(int) == shell_global)
    t_res_shell = t_res_arr[mask_res]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Courbes prédiction / exacte
    ax.plot(t_axis, U_pred[:, i], label=f'Prédiction u{i}',
            color='steelblue', linewidth=1.5, zorder=2)
    ax.plot(t_axis, U_exa[:, i],  label=f'Exact u{i}',
            color='darkorange', linewidth=1.5, linestyle='--', zorder=2)

    # Points d'observation (scatter sur la courbe exacte)
    if mask_obs.any():
        ax.scatter(
            t_obs_shell,
            u_obs_shell,
            color='green', marker='o', s=12, alpha=0.6, zorder=3,
            label=f'Observations ({mask_obs.sum()}pts)'
        )

    # Points de résidu (scatter en bas du graphe via axvline-like hack)
    if mask_res.any():
        y_min, y_max = ax.get_ylim()
        rug_y = np.full(mask_res.sum(), y_min + 0.02 * (y_max - y_min))
        ax.scatter(
            t_res_shell,
            rug_y,
            color='red', marker='|', s=40, alpha=0.5, zorder=3,
            label=f'Points résidu ({mask_res.sum()}pts)'
        )

    ax.set_xlabel('Temps normalisé')
    ax.set_ylabel(r'$\tilde{u}(k,t)$')
    ax.set_title(f'Shell {shell_global}  —  Prédiction vs Exact')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    fig.savefig(PATH + f"/prediction_u{i}.png", dpi=150)
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT VUE GLOBALE  (carte 2D observation + résidu)
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.scatter(t_obs_arr, k_obs_arr, s=4, alpha=0.3, color='green', rasterized=True)
ax.set_xlabel('Temps normalisé')
ax.set_ylabel('Indice de shell k')
ax.set_title(f'Points d\'observation ({len(obs_ds)} pts)')

ax = axes[1]
ax.scatter(t_res_arr, k_res_arr, s=4, alpha=0.3, color='red', rasterized=True)
ax.set_xlabel('Temps normalisé')
ax.set_ylabel('Indice de shell k')
ax.set_title(f'Points de résidu ({len(res_ds)} pts)')

fig.tight_layout()
fig.savefig(PATH + "/points_obs_residus.png", dpi=150)
plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT LOSSES
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(Total_loss[0], label='Total Loss',       color='black')
ax.semilogy(Total_loss[1], label='Observation Loss',  color='green')
ax.semilogy(Total_loss[2], label='Physics Loss',      color='red')
ax.set_xlabel('Itérations')
ax.set_ylabel('Loss (log)')
ax.set_title('Évolution des losses')
ax.legend()
fig.tight_layout()
fig.savefig(PATH + "/losses.png", dpi=150)
plt.close(fig)
