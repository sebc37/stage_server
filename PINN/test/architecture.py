import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

class GOY_PINN(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, n_layers, n_fourier=256, sigma=1.0):
        super().__init__()
        self.B_fourier = torch.randn(n_input, n_fourier).to(device) * sigma
        fourier_out_dim = 2 * n_fourier
        activation = nn.Tanh
        self.input_layer = nn.Sequential(
            nn.Linear(fourier_out_dim, n_hidden),
            activation()
        )
        self.hidden_layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                activation()
            ) for _ in range(n_layers - 1)
        ])
        self.output_layer = nn.Linear(n_hidden, n_output)
        self.apply(init_weights)

    def fourier_embed(self, x):
        x_proj = 2 * torch.pi * x @ self.B_fourier
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, x):
        x = self.fourier_embed(x)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET DES OBSERVATIONS  (tirage uniforme sur la grille (k,t,u) — FIXE)
# ─────────────────────────────────────────────────────────────────────────────
class observation_dataset(Dataset):
    """
    Points d'observation tirés UNIFORMÉMENT et UNE SEULE FOIS sur la grille
    (k, t, u).  Ils restent fixes pendant tout l'entraînement.

    Coordonnées stockées :
        col 0 : indice de shell  k  (float, non normalisé)
        col 1 : temps normalisé  t  in [0, 1]
        col 2 : valeur           u
    """
    def __init__(self, Data_shell, k_min, k_max, Npts, ratio, seed=42):
        super().__init__()
        np.random.seed(seed)

        self.k_min = k_min
        self.k_max = k_max
        self.Npts  = Npts

        # Grille complète (indice temporel, indice de colonne dans Data_shell)
        k_indices = np.arange(2 * k_min, 2 * k_max)
        t_indices = np.arange(Npts)
        all_t, all_k = np.meshgrid(t_indices, k_indices, indexing='ij')
        all_t = all_t.flatten()
        all_k = all_k.flatten()

        N_total  = len(all_t)
        N_sample = max(1, int(ratio * N_total))
        chosen   = np.random.choice(N_total, size=N_sample, replace=False)

        sampled_t = all_t[chosen]
        sampled_k = all_k[chosen]
        sampled_u = Data_shell[sampled_t, sampled_k]

        t_norm = sampled_t.astype(np.float32) / float(Npts)

        self.tensor_data = torch.tensor(
            np.stack([
                sampled_k.astype(np.float32),
                t_norm,
                sampled_u.astype(np.float32)
            ], axis=1),
            dtype=torch.float32
        )

        # Sauvegarde des coordonnées brutes pour les plots
        self.k_obs = sampled_k.astype(np.float32)          # indice shell
        self.t_obs = t_norm                                  # temps normalisé

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        # retourne (k, t_norm, u)
        return self.tensor_data[idx, 0], self.tensor_data[idx, 1], self.tensor_data[idx, 2]


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET DES RÉSIDUS  (tirage aléatoire sur la grille (k,t) — FIXE)
# ─────────────────────────────────────────────────────────────────────────────
class residual_dataset(Dataset):
    """
    Points de collocation pour le résidu de la PDE GOY, tirés UNE SEULE FOIS
    avant l'entraînement et fixes pendant tout l'entraînement.

    Coordonnées stockées :
        col 0 : indice de shell  k  (entier stocké en float)
        col 1 : temps normalisé  t  in [0, 1]
    """
    def __init__(self, k_min_phy, k_max_phy, Npts, N_residual, seed=123):
        super().__init__()
        np.random.seed(seed)

        self.k_min_phy  = k_min_phy
        self.k_max_phy  = k_max_phy
        self.Npts       = Npts
        self.N_residual = N_residual

        # Tirage uniforme des shells (entiers)
        k_res = np.random.randint(k_min_phy, k_max_phy, size=N_residual).astype(np.float32)
        # Tirage uniforme des temps dans [0, 1]
        t_res = np.random.uniform(0.0, 1.0, size=N_residual).astype(np.float32)

        self.tensor_data = torch.tensor(
            np.stack([k_res, t_res], axis=1),
            dtype=torch.float32
        )

        # Sauvegarde pour les plots
        self.k_res = k_res
        self.t_res = t_res

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx, 0], self.tensor_data[idx, 1]


# ─────────────────────────────────────────────────────────────────────────────
#  ANCIENS DATASETS  (conservés pour compatibilité)
# ─────────────────────────────────────────────────────────────────────────────
class initials_variables_data(Dataset):
    def __init__(self, X_ic, nbr_initial_t, k_min, k_max, t_0):
        self.k_min = k_min
        self.k_max = k_max
        self.X_ic  = torch.from_numpy(X_ic[self.k_min:self.k_max * 2])
        self.nbr_initial_t = nbr_initial_t

        self.x_initial = np.array([k for k in range(k_min, k_max * 2)], dtype="float32")
        self.t_initial = np.array([t_0 for _ in range(self.nbr_initial_t)], dtype="float32")

        self.tensor_data_ic = torch.tensor(self.X_ic, dtype=torch.float32)
        self.grille = np.meshgrid(self.x_initial, self.t_initial)
        self.grille = torch.tensor(self.grille, dtype=torch.float32).T.view(
            np.shape(self.x_initial)[0] * np.shape(self.t_initial)[0], 2
        )
        self.tensor_data = torch.ones(
            (np.shape(self.x_initial)[0] * np.shape(self.t_initial)[0], 3), dtype=torch.float32
        )
        self.tensor_data[:, 0:2] = self.grille
        self.tensor_data[:, 2]   = self.tensor_data_ic

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        return self.tensor_data[idx, 0], self.tensor_data[idx, 1], self.tensor_data[idx, 2]


class boundary_variables_data(Dataset):
    def __init__(self, X_boundary, Npts, time, f, dt):
        self.Npts = Npts
        nb_k = X_boundary.shape[1]
        t_axis = np.linspace(0, 1, Npts, dtype=np.float32)
        k_axis = np.arange(nb_k, dtype=np.float32)
        all_t, all_k = np.meshgrid(t_axis, k_axis, indexing='ij')
        all_u = X_boundary
        self.tensor_data_bc = torch.tensor(
            np.stack([all_k.flatten(), all_t.flatten(), all_u.flatten()], axis=1),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.tensor_data_bc)

    def __getitem__(self, idx):
        return self.tensor_data_bc[idx, 0], self.tensor_data_bc[idx, 1], self.tensor_data_bc[idx, 2]


class colocations_variables_data(Dataset):
    def __init__(self, Data_train):
        self.nb_colocation_pnt = len(Data_train[0])
        k_vals = np.array(Data_train[0], dtype=np.float32)
        t_vals = np.array(Data_train[1], dtype=np.float32)
        u_vals = np.array(Data_train[2], dtype=np.float32)
        self.tensor_data_colocation = torch.tensor(
            np.stack([k_vals, t_vals, u_vals], axis=1), dtype=torch.float32
        )

    def __len__(self):
        return self.nb_colocation_pnt

    def __getitem__(self, idx):
        return (
            self.tensor_data_colocation[idx, 0],
            self.tensor_data_colocation[idx, 1],
            self.tensor_data_colocation[idx, 2],
        )


class grid_data(Dataset):
    def __init__(self, k_min, k_max, t_min, t_max, Npts):
        self.k_min = k_min
        self.k_max = k_max
        self.Npts  = Npts
        x = np.arange(k_min, 2 * k_max, dtype=np.float32)
        t = np.linspace(t_min, t_max, Npts, dtype=np.float32)
        self.x = x
        self.t = t
        self.N_k = len(x)
        self.N_t = len(t)
        all_k, all_t = np.meshgrid(x, t, indexing='ij')
        self.grid = torch.tensor(
            np.stack([all_k.flatten(), all_t.flatten()], axis=1), dtype=torch.float32
        )

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        return self.grid[idx, 0], self.grid[idx, 1]
