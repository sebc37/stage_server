"""
==========================================================================================
ANaGRAM (Algorithme 2) applique au PINN du modele de shell turbulent (GOY)
==========================================================================================

Ce script adapte l'entrainement par gradient naturel tronque (ANaGRAM) au PINN
"pinn_lightning_shells_fullplot.py" fourni : meme architecture de reseau (Fourier
embedding + couches SIREN), meme physique (modele de shell GOY complexe couple),
mais remplace Adam/LBFGS (PyTorch Lightning) par la boucle de l'Algorithme 2.

Difference cle par rapport a l'ANaGRAM "vanilla" (2 operateurs D, B) : ici la loss
a TROIS morceaux :

    - loss_ic  : condition initiale     ->  operateur "IC"   :  u_theta(k, t0) - u0
    - loss_obs : observations (donnees) ->  operateur "obs"  :  u_theta(k_obs, t_obs) - u_obs
    - loss_pde : residu physique GOY    ->  operateur "PDE"  :  dU_k/dt - RHS_k(U)

L'algorithme se generalise directement : on empile les TROIS jacobiennes (par
rapport a theta) au lieu de 2 (D et B), chacune ponderee par sa loss weight
(w_ic, w_obs, w_pde) divisee par son nombre de points -- exactement l'equivalent
"multi-operateurs" de phi_hat = Jac(D[u], B[u]) mais avec un troisieme bloc.

    phi_hat_theta = Jac_theta [ sqrt(w_pde/n_pde) * r_pde(theta) ;
                                 sqrt(w_obs/n_obs) * r_obs(theta) ;
                                 sqrt(w_ic /n_ic ) * r_ic (theta) ]

Le reste de l'algorithme (SVD tronquee, pas de Gauss-Newton, line search sur eta)
est identique a l'implementation "vanilla" precedente.

Point technique : la derivee temporelle dU/dt (necessaire au residu PDE) est
calculee fonctionnellement via `torch.func.jacfwd` (mode direct, peu couteux car
t est un scalaire par point), imbriquee DANS le `jacrev` externe par rapport a
theta (mode inverse, adapte a un grand nombre de parametres). Ce pattern
"forward-over-reverse" est le meme que celui utilise pour les hessiennes et est
bien supporte par torch.func.

Requiert torch >= 2.0 (torch.func : functional_call, vmap, jacrev, jacfwd).
==========================================================================================
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable, Optional
import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, vmap, jacrev, jacfwd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# ==========================================================================================
# 1. ARCHITECTURE DU RESEAU -- identique au script d'origine (Fourier + SIREN)
# ==========================================================================================

class FourierEmbedding(nn.Module):
    def __init__(self, in_features: int, mapping_size: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_features, mapping_size) * scale
        self.register_buffer("B", B)
        self.out_features = mapping_size * 2

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w0: float = 1.0, is_first: bool = False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(in_features, is_first)

    def _init_weights(self, in_features, is_first):
        with torch.no_grad():
            bound = (1.0 / in_features) if is_first else (math.sqrt(6.0 / in_features) / self.w0)
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))


class PINN(nn.Module):
    def __init__(self, layers, w0: float = 30.0,
                 fourier_mapping_size: int = 64, fourier_scale: float = 10.0):
        super().__init__()
        assert layers[-1] == 2, "la derniere couche doit avoir 2 sorties (Re, Im)"
        in_features = layers[0]
        self.fourier = FourierEmbedding(in_features, mapping_size=fourier_mapping_size, scale=fourier_scale)
        dims = [self.fourier.out_features] + list(layers[1:])
        modules = []
        n_layers = len(dims) - 1
        for i in range(n_layers):
            is_last = (i == n_layers - 1)
            is_first = (i == 0)
            if is_last:
                modules.append(nn.Linear(dims[i], dims[i + 1]))
            else:
                wi = w0 if is_first else 1.0
                modules.append(SineLayer(dims[i], dims[i + 1], w0=wi, is_first=is_first))
        self.net = nn.Sequential(*modules)

    def forward(self, k, t):
        x = torch.cat([k, t], dim=1)
        x = self.fourier(x)
        out = self.net(x)
        return out[:, 0:1], out[:, 1:2]


def conj_prod(re_p, im_p, re_q, im_q):
    """conj(U_p) * conj(U_q)"""
    re = re_p * re_q - im_p * im_q
    im = -(re_p * im_q + im_p * re_q)
    return re, im


# ==========================================================================================
# 2. UTILITAIRES THETA <-> PARAMS (vecteur plat requis par jacrev/SVD)
# ==========================================================================================

def params_dict(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.named_parameters()} # dictionnaire des paramètres du modèles


def flatten_params(params: dict) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params.values()]) # tensor des paramètres du modèle applatis


def unflatten_params(theta: torch.Tensor, ref: dict) -> dict: # permet de reconstruire le dictionnaire après flatten_params
    out, i = {}, 0
    for k, p in ref.items():
        n = p.numel()
        out[k] = theta[i:i + n].reshape(p.shape)
        i += n
    return out


# ==========================================================================================
# 3. OPERATEURS FONCTIONNELS : u_theta(k,t), du/dt, residu PDE (GOY), residu obs/IC
# ==========================================================================================

def make_functional_ops(model: nn.Module, ref_params: dict, O_k: torch.Tensor, a: float, b: float, nu: float):
    """Construit les fonctions purement fonctionnelles de theta necessaires a ANaGRAM :
    u_pair_batch (valeur), du_dt_batch (derivee temporelle), pde_residual_theta,
    value_residual_theta (reutilise pour IC et pour les observations)."""

    def u_pair(theta: torch.Tensor, k_s: torch.Tensor, t_s: torch.Tensor) -> torch.Tensor: # utilise les paramètre flatten et le modèle (nn) pour créer une version fonctionnelle du modèle
        p = unflatten_params(theta, ref_params)
        re, im = functional_call(model, p, (k_s.reshape(1, 1), t_s.reshape(1, 1)))
        return torch.stack([re.reshape(()), im.reshape(())])   # (2,) = [Re, Im]

    u_pair_batch = vmap(u_pair, in_dims=(None, 0, 0))                     # (N,2)
    du_dt_pair = jacfwd(u_pair, argnums=2)                                # d[Re,Im]/dt, mode direct (t scalaire)
    du_dt_batch = vmap(du_dt_pair, in_dims=(None, 0, 0))                  # (N,2)

    def value_residual_theta(theta, k_flat, t_flat, u_re_target, u_im_target):
        """Residu de VALEUR (utilise pour IC et pour les observations) :
        u_theta(k,t) - u_cible, aplati en un seul vecteur [re...; im...]."""
        uv = u_pair_batch(theta, k_flat, t_flat)               # (N,2)
        res_re = uv[:, 0] - u_re_target
        res_im = uv[:, 1] - u_im_target
        return torch.cat([res_re, res_im])

    def pde_residual_theta(theta, k_flat, t_flat, B_t: int, n_shells: int):
        """Residu physique du modele GOY, vectorise sur (B_t, n_shells) --
        version fonctionnelle (theta) de PINNModule.pde_residual."""
        uv = u_pair_batch(theta, k_flat, t_flat)               # (B_t*n_shells, 2)
        duv = du_dt_batch(theta, k_flat, t_flat)               # (B_t*n_shells, 2)

        Ure = uv[:, 0].reshape(B_t, n_shells)
        Uim = uv[:, 1].reshape(B_t, n_shells)
        dUre_dt = duv[:, 0].reshape(B_t, n_shells)
        dUim_dt = duv[:, 1].reshape(B_t, n_shells)

        Ure_p = F.pad(Ure, (2, 2))
        Uim_p = F.pad(Uim, (2, 2))

        def shell(offset):
            s = 2 + offset
            return Ure_p[:, s:s + n_shells], Uim_p[:, s:s + n_shells]

        Ure_km2, Uim_km2 = shell(-2)
        Ure_km1, Uim_km1 = shell(-1)
        Ure_kp1, Uim_kp1 = shell(+1)
        Ure_kp2, Uim_kp2 = shell(+2)

        A_re, A_im = conj_prod(Ure_kp1, Uim_kp1, Ure_kp2, Uim_kp2)
        Bt_re, Bt_im = conj_prod(Ure_km1, Uim_km1, Ure_kp1, Uim_kp1)
        C_re, C_im = conj_prod(Ure_km2, Uim_km2, Ure_km1, Uim_km1)

        N_re = A_re - a * Bt_re + b * C_re
        N_im = A_im - a * Bt_im + b * C_im

        RHS_re = -O_k * N_im - nu * (O_k ** 2) * Ure
        RHS_im = O_k * N_re - nu * (O_k ** 2) * Uim

        res_re = dUre_dt - RHS_re
        res_im = dUim_dt - RHS_im

        scale = O_k ** 2
        res_re_n = res_re / scale
        res_im_n = res_im / scale
        return torch.cat([res_re_n.reshape(-1), res_im_n.reshape(-1)])

    return u_pair_batch, pde_residual_theta, value_residual_theta


# ==========================================================================================
# 4. LINE SEARCH (recherche du nombre d'or, identique a la version "vanilla")
# ==========================================================================================

def golden_section_search(loss_fn: Callable[[float], float], a: float = 0.0,
                           b: float = 2.0, tol: float = 1e-5, max_iter: int = 60) -> float:
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = loss_fn(c)[0], loss_fn(d)[0]
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = loss_fn(c)[0]
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = loss_fn(d)[0]
    return (a + b) / 2.0


# ==========================================================================================
# 5. ANaGRAM MULTI-OPERATEURS (IC + obs + PDE) -- generalisation de l'algorithme 2
# ==========================================================================================

@dataclass
class GOYData:
    """Tenseurs de donnees necessaires a chaque bloc de residu."""
    k_values: torch.Tensor          # (n_shells,)
    t_values: torch.Tensor          # (n_t,)
    U_re: torch.Tensor              # (n_t, n_shells)
    U_im: torch.Tensor              # (n_t, n_shells)
    k_obs_idx: torch.Tensor         # (n_shells_obs,) indices dans k_values
    idx_t0: int = 0                 # index du temps initial


@dataclass
class ANaGRAMConfigGOY:
    epsilon: float = 1e-4            # seuil de troncature (relatif au sigma max)
    eta_max: float = 1.0             # borne sup pour la line search
    n_iter: int = 200
    w_ic: float = 1.0
    w_obs: float = 1.0
    w_pde: float = 1.0
    n_t_colloc: int = 20             # nb de pas de temps tires pour le batch PDE (x n_shells points)
    n_t_obs: int = 20                # nb de pas de temps tires pour le batch observations
    pde_start_iter: int = 0          # equivalent de pde_start_epoch : PDE desactivee avant cette iteration
    seed: Optional[int] = None


def build_batches(data: GOYData, cfg: ANaGRAMConfigGOY, iteration: int, device, generator=None):
    """Tire aleatoirement les batches (colloc, obs) et construit le batch IC fixe.
    Reproduit la structure de CollocationDataset / ObservationDataset / build_ic_tensors
    du script d'origine, mais directement sous forme de tenseurs (pas de DataLoader) --
    plus pratique pour piloter manuellement les iterations d'ANaGRAM."""

    n_shells = data.k_values.shape[0]
    n_t = data.t_values.shape[0]

    # --- collocation (PDE) : n_t_colloc pas de temps, TOUS les shells ---
    idx_c = torch.randint(0, n_t, (cfg.n_t_colloc,), generator=generator)
    t_c = data.t_values[idx_c]                                          # (B_t,)
    k_grid = data.k_values.unsqueeze(0).expand(cfg.n_t_colloc, n_shells)
    t_grid = t_c.unsqueeze(1).expand(cfg.n_t_colloc, n_shells)
    k_c_flat = k_grid.reshape(-1).to(device)
    t_c_flat = t_grid.reshape(-1).to(device)

    # --- observations : n_t_obs pas de temps, uniquement les shells observes ---
    k_obs = data.k_values[data.k_obs_idx]                               # (n_shells_obs,)
    n_shells_obs = k_obs.shape[0]
    idx_o = torch.randint(0, n_t, (cfg.n_t_obs,), generator=generator)
    t_o = data.t_values[idx_o]
    k_o_grid = k_obs.unsqueeze(0).expand(cfg.n_t_obs, n_shells_obs)
    t_o_grid = t_o.unsqueeze(1).expand(cfg.n_t_obs, n_shells_obs)
    u_o_re = data.U_re[idx_o][:, data.k_obs_idx]
    u_o_im = data.U_im[idx_o][:, data.k_obs_idx]
    k_o_flat = k_o_grid.reshape(-1).to(device)
    t_o_flat = t_o_grid.reshape(-1).to(device)
    u_o_re_flat = u_o_re.reshape(-1).to(device)
    u_o_im_flat = u_o_im.reshape(-1).to(device)

    # --- IC : t0 fixe, TOUS les shells (fixe pour toutes les iterations) ---
    k0 = data.k_values.to(device)
    t0 = torch.full_like(k0, data.t_values[data.idx_t0].item())
    u0_re = data.U_re[data.idx_t0].to(device)
    u0_im = data.U_im[data.idx_t0].to(device)

    return dict(
        colloc=(k_c_flat, t_c_flat, cfg.n_t_colloc, n_shells),
        obs=(k_o_flat, t_o_flat, u_o_re_flat, u_o_im_flat),
        ic=(k0, t0, u0_re, u0_im),
    ), idx_c, idx_o


def anagram_step_goy(theta: torch.Tensor, ops, batches, cfg: ANaGRAMConfigGOY, w_pde_now: float):
    """Une iteration d'ANaGRAM multi-operateurs (lignes 2 a 8 de l'algorithme 2,
    generalisees a 3 blocs de residus : PDE, obs, IC)."""
    u_pair_batch, pde_residual_theta, value_residual_theta = ops

    k_c_flat, t_c_flat, B_t_c, n_shells = batches["colloc"]
    k_o_flat, t_o_flat, u_o_re, u_o_im = batches["obs"]
    k0, t0, u0_re, u0_im = batches["ic"]

    n_pde = 2 * B_t_c * n_shells
    n_obs = 2 * k_o_flat.shape[0]
    n_ic = 2 * k0.shape[0]

    s_pde = math.sqrt(w_pde_now / n_pde) if w_pde_now > 0 else 0.0
    s_obs = math.sqrt(cfg.w_obs / n_obs)
    s_ic = math.sqrt(cfg.w_ic / n_ic)

    # --- ligne 2 : phi_hat = jacobienne empilee (PDE ; obs ; IC) par rapport a theta ---
    def stacked_residual(th):
        parts = []
        if w_pde_now > 0:
            parts.append(s_pde * pde_residual_theta(th, k_c_flat, t_c_flat, B_t_c, n_shells))
        parts.append(s_obs * value_residual_theta(th, k_o_flat, t_o_flat, u_o_re, u_o_im))
        parts.append(s_ic * value_residual_theta(th, k0, t0, u0_re, u0_im))
        return torch.cat(parts)

    phi_hat = jacrev(stacked_residual)(theta)                 # (N, P)
    r = stacked_residual(theta).detach()                      # (N,)

    # --- ligne 3 : SVD ---
    U, S, Vh = torch.linalg.svd(phi_hat, full_matrices=False)
    V = Vh.T

    # --- ligne 4 : troncature ---
    threshold = cfg.epsilon * S.max()
    S_pinv = torch.where(S > threshold, 1.0 / S, torch.zeros_like(S))

    # --- lignes 5-6 : pas de Gauss-Newton tronque ---
    d_theta = V @ (S_pinv * (U.T @ r))

    # --- ligne 7 : line search sur eta (loss totale ponderee, meme convention que
    #     PINNModule.compute_losses : loss = w_ic*loss_ic + w_obs*loss_obs + w_pde*loss_pde) ---
    def total_loss(eta: float) -> float:
        th_new = theta - eta * d_theta
        with torch.no_grad():
            r_obs = value_residual_theta(th_new, k_o_flat, t_o_flat, u_o_re, u_o_im)
            r_ic = value_residual_theta(th_new, k0, t0, u0_re, u0_im)
            loss = cfg.w_obs * torch.mean(r_obs ** 2) + cfg.w_ic * torch.mean(r_ic ** 2)
            if w_pde_now > 0:
                r_pde = pde_residual_theta(th_new, k_c_flat, t_c_flat, B_t_c, n_shells)
                loss = loss + w_pde_now * torch.mean(r_pde ** 2)
        return loss.item(), cfg.w_obs * torch.mean(r_obs ** 2).item(), cfg.w_ic * torch.mean(r_ic ** 2).item(),w_pde_now * torch.mean(r_pde ** 2).item() #ici

    eta_star = golden_section_search(total_loss, a=0.0, b=cfg.eta_max)
    theta_new = theta - eta_star * d_theta
    loss_val = total_loss(eta_star)
    rank_eff = (S > threshold).sum().item()
    return theta_new, loss_val[0], eta_star, rank_eff, S.shape[0], loss_val[1], loss_val[2], loss_val[3]


def train_anagram_goy(model: nn.Module, data: GOYData, a: float, b: float, nu: float,
                       cfg: ANaGRAMConfigGOY, device: str = device, verbose_setup: bool = True):
    model.to(device)
    ref_params = params_dict(model)
    theta = flatten_params(ref_params).to(device)

    n_shells = data.k_values.shape[0]
    O_k = 0.125 * 2.0 ** torch.arange(1, n_shells + 1, dtype=torch.float32, device=device)
    ops = make_functional_ops(model, ref_params, O_k, a, b, nu)

    if verbose_setup:
        k_obs_vals = data.k_values[data.k_obs_idx].tolist()
        print("=" * 70)
        print(f"Shells observes (indices 0-based)   : {data.k_obs_idx.tolist()}")
        print(f"Shells observes (valeurs de k)       : {k_obs_vals}")
        print(f"Nombre total de shells                : {n_shells}")
        print(f"Plage temporelle disponible            : [{data.t_values.min().item():.4g}, "
              f"{data.t_values.max().item():.4g}]  ({data.t_values.shape[0]} pas de temps)")
        print(f"t0 (condition initiale)                : {data.t_values[data.idx_t0].item():.4g}")
        print(f"Points PDE par iteration                : {cfg.n_t_colloc} pas de temps x {n_shells} shells")
        print(f"Points observation par iteration        : {cfg.n_t_obs} pas de temps x "
              f"{len(data.k_obs_idx)} shells observes")
        print("=" * 70)

    generator = torch.Generator().manual_seed(cfg.seed) if cfg.seed is not None else None
    history = []
    history_obs = []
    history_ic = []
    history_phy = []
    idx_c_history, idx_o_history = [], []

    for it in tqdm.tqdm(range(cfg.n_iter)):
        batches, idx_c, idx_o = build_batches(data, cfg, it, device, generator=generator)
        idx_c_history.append(idx_c.cpu().numpy())
        idx_o_history.append(idx_o.cpu().numpy())
        w_pde_now = cfg.w_pde if it >= cfg.pde_start_iter else 0.0

        theta, loss_val, eta_star, rk, rt ,obs,ic,phy = anagram_step_goy(theta, ops, batches, cfg, w_pde_now) #ici
        history.append(loss_val) #ici
        history_obs.append(obs)
        history_ic.append(ic)
        history_phy.append(phy)
        if it % 5 == 0 or it == cfg.n_iter - 1:
            print(f"iter {it:4d} | loss={loss_val:.4e} | eta={eta_star:.4f} "
                  f"| rang eff. {rk}/{rt} | w_pde={w_pde_now:.2f} | loss_obs={obs:.4e} | loss_ic={ic:.4e} | loss_phy={phy:.4e}") #ici

    final_params = unflatten_params(theta, ref_params)
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(final_params[k])

    return model, history, idx_c_history, idx_o_history, history_obs, history_ic, history_phy #ici


# ==========================================================================================
# 6. FONCTIONS DE VISUALISATION -- equivalents des fonctions du script Lightning original,
#    mais adaptees a un modele "fonctionnel" pilote par un vecteur theta plutot que par un
#    PINNModule Lightning (pas de .eval()/.train(), pas de forward via nn.Module direct).
# ==========================================================================================

@torch.no_grad()
def predict_full_grid(u_pair_batch, theta: torch.Tensor, k_values: torch.Tensor,
                       t_values: torch.Tensor, chunk_size: int = 2000, device: str = device):
    """Prediction (Re, Im) sur TOUTE la grille (tous les shells, tous les t), par blocs
    de pas de temps (chunk_size) pour eviter de charger n_t x n_shells points d'un coup."""
    n_shells = k_values.shape[0]
    n_t = t_values.shape[0]
    u_re_pred_full = torch.empty(n_t, n_shells)
    u_im_pred_full = torch.empty(n_t, n_shells)

    for start in range(0, n_t, chunk_size):
        end = min(start + chunk_size, n_t)
        t_chunk = t_values[start:end]
        B = t_chunk.shape[0]
        k_grid = k_values.unsqueeze(0).expand(B, n_shells)
        t_grid = t_chunk.unsqueeze(1).expand(B, n_shells)
        k_flat = k_grid.reshape(-1).to(device)
        t_flat = t_grid.reshape(-1).to(device)

        uv = u_pair_batch(theta, k_flat, t_flat)          # (B*n_shells, 2)
        u_re_pred_full[start:end] = uv[:, 0].reshape(B, n_shells).cpu()
        u_im_pred_full[start:end] = uv[:, 1].reshape(B, n_shells).cpu()

    return u_re_pred_full, u_im_pred_full


@torch.no_grad()
def plot_full_grid_predictions_anagram(u_pair_batch, theta, k_values, t_values, U_re, U_im,
                                        shells_to_plot=None, chunk_size: int = 2000,
                                        device: str = device, max_cols: int = 2):
    """Trace prediction vs verite pour TOUS les shells (ou une liste donnee), Re et Im
    cote a cote -- equivalent de plot_full_grid_predictions, mais par defaut sur TOUS
    les shells (shells_to_plot=None) au lieu d'un sous-ensemble de 4."""
    n_shells = k_values.shape[0]
    if shells_to_plot is None:
        shells_to_plot = list(range(n_shells))

    u_re_pred_full, u_im_pred_full = predict_full_grid(
        u_pair_batch, theta, k_values, t_values, chunk_size=chunk_size, device=device)

    n_plot = len(shells_to_plot)
    t_np = t_values.cpu()
    fig, axes = plt.subplots(n_plot, 2, figsize=(11, 2.6 * n_plot), squeeze=False)

    for row, s in enumerate(shells_to_plot):
        axes[row, 0].plot(t_np, U_re[:, s].cpu(), lw=1, label="verite", alpha=0.7)
        axes[row, 0].plot(t_np, u_re_pred_full[:, s], lw=1, label="prediction")
        axes[row, 0].set_title(f"Re(U), shell k={k_values[s].item():.0f}")
        axes[row, 0].set_xlabel("t")
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].plot(t_np, U_im[:, s].cpu(), lw=1, label="verite", alpha=0.7)
        axes[row, 1].plot(t_np, u_im_pred_full[:, s], lw=1, label="prediction")
        axes[row, 1].set_title(f"Im(U), shell k={k_values[s].item():.0f}")
        axes[row, 1].set_xlabel("t")
        axes[row, 1].legend(fontsize=8)

    fig.tight_layout()
    return fig, u_re_pred_full, u_im_pred_full


@torch.no_grad()
def plot_rmse_module_per_shell_anagram(u_pair_batch, theta, k_values, t_values, U_re, U_im,
                                        chunk_size: int = 2000, device: str = device):
    """RMSE du module au carre |U|^2, par shell -- equivalent de plot_rmse_module_per_shell."""
    u_re_pred_full, u_im_pred_full = predict_full_grid(
        u_pair_batch, theta, k_values, t_values, chunk_size=chunk_size, device=device)

    mod2_pred = torch.sqrt(u_re_pred_full ** 2 + u_im_pred_full ** 2)
    mod2_exa = torch.sqrt(U_re.cpu() ** 2 + U_im.cpu() ** 2)
    diff = mod2_pred - mod2_exa
    rmse_per_shell = torch.sqrt(torch.mean(diff ** 2, dim=0))    # (n_shells,)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values.cpu(), rmse_per_shell, marker="o")
    ax.set_yscale("log")
    ax.set_xlabel("shell k")
    ax.set_ylabel("RMSE(|U|²) [echelle log]")
    ax.set_title("RMSE du module au carre, par shell (ANaGRAM)")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig, rmse_per_shell


@torch.no_grad()
def plot_energy_spectrum_per_shell_anagram(u_pair_batch, theta, k_values, t_values, U_re, U_im,
                                            chunk_size: int = 2000, device: str = device):
    """log( mean_t(|U|^2) ) par shell, prediction vs verite -- equivalent de
    plot_energy_spectrum_per_shell."""
    u_re_pred_full, u_im_pred_full = predict_full_grid(
        u_pair_batch, theta, k_values, t_values, chunk_size=chunk_size, device=device)

    E_pred = torch.mean(u_re_pred_full ** 2 + u_im_pred_full ** 2, dim=0)
    E_exa = torch.mean(U_re.cpu() ** 2 + U_im.cpu() ** 2, dim=0)
    log_E_pred = torch.log(E_pred)
    log_E_exa = torch.log(E_exa)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values.cpu(), log_E_exa, marker="o", label="realite")
    ax.plot(k_values.cpu(), log_E_pred, marker="x", label="prediction")
    ax.set_xlabel("shell k")
    ax.set_ylabel("log( mean_t(|U|²) )")
    ax.set_title("Spectre energetique par shell (ANaGRAM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, log_E_pred, log_E_exa


@torch.no_grad()
def plot_kurtosis_per_shell_anagram(u_pair_batch, theta, k_values, t_values, U_re, U_im,
                                     chunk_size: int = 2000, device: str = device):
    """log( kurtosis(|U|) ) par shell, prediction vs verite -- equivalent de
    plot_kurtosis_per_shell."""
    u_re_pred_full, u_im_pred_full = predict_full_grid(
        u_pair_batch, theta, k_values, t_values, chunk_size=chunk_size, device=device)

    def kurtosis(u_re, u_im):
        mod2 = u_re ** 2 + u_im ** 2
        mod4 = mod2 ** 2
        mean_mod2 = torch.mean(mod2, dim=0)
        mean_mod4 = torch.mean(mod4, dim=0)
        return mean_mod4 / (mean_mod2 ** 2)

    kurt_pred = kurtosis(u_re_pred_full, u_im_pred_full)
    kurt_exa = kurtosis(U_re.cpu(), U_im.cpu())
    log_kurt_pred = torch.log(kurt_pred)
    log_kurt_exa = torch.log(kurt_exa)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values.cpu(), log_kurt_exa, marker="o", label="realite")
    ax.plot(k_values.cpu(), log_kurt_pred, marker="x", label="prediction")
    ax.set_xlabel("shell k")
    ax.set_ylabel("log( kurtosis(|U|) )")
    ax.set_title("Kurtosis du module par shell (ANaGRAM)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, log_kurt_pred, log_kurt_exa


# ==========================================================================================
# 6bis. VERIFICATION / VISUALISATION DES BATCHES UTILISES PENDANT L'ENTRAINEMENT
# ==========================================================================================
#
# Objectif : pouvoir VERIFIER visuellement, apres coup, que (1) les shells observes sont
# bien ceux voulus, et (2) que les batches PDE et observation tires a chaque iteration
# couvrent bien l'ENSEMBLE de la plage temporelle disponible (et pas seulement une
# sous-fenetre), puisque contrairement a l'entrainement Adam/LBFGS original (qui voit
# potentiellement tous les points a chaque epoch via CombinedLoader), ANaGRAM ne tire
# qu'un sous-batch de pas de temps a chaque iteration (cf. n_t_colloc / n_t_obs).

@torch.no_grad()
def plot_batch_snapshot(data: GOYData, cfg: ANaGRAMConfigGOY, device: str = device):
    """Tire UN batch (comme le ferait une iteration d'ANaGRAM) et affiche dans le plan
    (t, k) les points utilises par chaque bloc de residu : collocation PDE (tous les
    shells), observations (uniquement les shells choisis), IC (t0 fixe, tous les shells).
    Permet de verifier d'un coup d'oeil la structure geometrique d'un batch."""
    batches, idx_c, idx_o = build_batches(data, cfg, iteration=0, device=device)
    k_c_flat, t_c_flat, B_t_c, n_shells = batches["colloc"]
    k_o_flat, t_o_flat, _, _ = batches["obs"]
    k0, t0, _, _ = batches["ic"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(t_c_flat.cpu(), k_c_flat.cpu(), s=18, alpha=0.45, color="tab:blue",
               label=f"collocation PDE ({B_t_c} pas de temps × {n_shells} shells)")
    ax.scatter(t_o_flat.cpu(), k_o_flat.cpu(), s=45, alpha=0.9, color="tab:orange", marker="x",
               label=f"observations ({cfg.n_t_obs} pas de temps × {len(data.k_obs_idx)} shells)")
    ax.scatter(t0.cpu(), k0.cpu(), s=70, alpha=0.9, color="tab:green", marker="^",
               label="condition initiale (IC, t0 fixe)")

    ax.set_xlabel("t")
    ax.set_ylabel("shell k")
    ax.set_yticks(data.k_values.tolist())
    ax.set_title("Instantane d'un batch ANaGRAM (une iteration)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_batch_time_coverage(idx_c_history, idx_o_history, t_values: torch.Tensor,
                              idx_t0: int = 0, n_bins: int = 60):
    """A partir des historiques d'indices reellement tires pendant TOUT l'entrainement
    (idx_c_history, idx_o_history -- une entree par iteration), verifie que l'ensemble
    de la plage temporelle a bien ete vue :
      - panneau haut  : nuage (iteration, t tire) pour PDE et observations
      - panneau bas   : histogramme cumule des t tires (doit couvrir toute la plage,
                        sans trou, si l'echantillonnage est bien uniforme)
    """
    t_np = t_values.cpu().numpy()
    n_iter = len(idx_c_history)

    it_c = np.concatenate([np.full(len(idx), i) for i, idx in enumerate(idx_c_history)])
    t_c_all = np.concatenate([t_np[idx] for idx in idx_c_history])
    it_o = np.concatenate([np.full(len(idx), i) for i, idx in enumerate(idx_o_history)])
    t_o_all = np.concatenate([t_np[idx] for idx in idx_o_history])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].scatter(it_c, t_c_all, s=3, alpha=0.25, color="tab:blue", label="collocation PDE")
    axes[0].scatter(it_o, t_o_all, s=3, alpha=0.25, color="tab:orange", label="observations")
    axes[0].axhline(t_np[idx_t0], color="tab:green", lw=1.5, ls="--", label="t0 (IC)")
    axes[0].set_ylabel("t echantillonne")
    axes[0].set_title(f"Couverture temporelle des batches sur les {n_iter} iterations d'entrainement")
    axes[0].legend(loc="upper right", fontsize=9)

    axes[1].hist(t_c_all, bins=n_bins, alpha=0.5, color="tab:blue", label="collocation PDE")
    axes[1].hist(t_o_all, bins=n_bins, alpha=0.5, color="tab:orange", label="observations")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("nb de tirages cumules")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.tight_layout()

    # -- petit rapport texte de couverture (min/max/fraction de la plage vue) --
    t_min, t_max = t_np.min(), t_np.max()
    span = t_max - t_min
    cov_c = (t_c_all.min() - t_min) / span, (t_max - t_c_all.max()) / span
    cov_o = (t_o_all.min() - t_min) / span, (t_max - t_o_all.max()) / span
    print(f"Couverture temporelle collocation PDE : [{t_c_all.min():.4g}, {t_c_all.max():.4g}] "
          f"sur plage totale [{t_min:.4g}, {t_max:.4g}]")
    print(f"Couverture temporelle observations    : [{t_o_all.min():.4g}, {t_o_all.max():.4g}] "
          f"sur plage totale [{t_min:.4g}, {t_max:.4g}]")

    return fig


# ==========================================================================================
# 7. EXEMPLE D'UTILISATION -- a adapter avec vos vraies donnees GOY
# ==========================================================================================
#
# Remplacement direct du bloc "5. SCRIPT PRINCIPAL" du fichier d'origine :
# au lieu de PINNDataModule + PINNModule + Trainer Lightning (Adam puis LBFGS),
# on construit un GOYData et on appelle train_anagram_goy.
#
# NB performance : ANaGRAM calcule une SVD de phi_hat, matrice (N x P) avec
# N = 2*(n_t_colloc*n_shells + n_t_obs*n_shells_obs + n_shells). Contrairement a
# Adam/LBFGS, il faut donc garder n_t_colloc et n_t_obs MODESTES (quelques dizaines
# de pas de temps par iteration, pas les 90090 d'un coup) -- c'est le batch (x_i^D),
# (x_i^B) de l'algorithme, pas l'integralite des donnees.
# ==========================================================================================

def _simulate_goy_reference(n_shells, n_t, a, b, nu, dt=1e-3, seed=0):
    """Genere des donnees synthetiques COHERENTES avec le modele GOY (integration
    explicite RK4), uniquement pour demontrer/tester le pipeline sans les vraies
    donnees data.dat. A remplacer par le chargement reel dans un cas d'usage."""
    rng = np.random.default_rng(seed)
    O_k = 0.125 * 2.0 ** np.arange(1, n_shells + 1)

    def rhs(U):
        # U : (n_shells,) complexe
        Up = np.pad(U, (2, 2))
        Ukp1 = Up[3:3 + n_shells]
        Ukp2 = Up[4:4 + n_shells]
        Ukm1 = Up[1:1 + n_shells]
        Ukm2 = Up[0:0 + n_shells]
        N = (np.conj(Ukp1) * np.conj(Ukp2)
             - a * np.conj(Ukm1) * np.conj(Ukp1)
             + b * np.conj(Ukm2) * np.conj(Ukm1))
        return 1j * O_k * N - nu * O_k ** 2 * U

    U = (rng.standard_normal(n_shells) + 1j * rng.standard_normal(n_shells)) * 0.1
    traj = np.empty((n_t, n_shells), dtype=complex)
    traj[0] = U
    for i in range(1, n_t):
        k1 = rhs(U)
        k2 = rhs(U + 0.5 * dt * k1)
        k3 = rhs(U + 0.5 * dt * k2)
        k4 = rhs(U + dt * k3)
        U = U + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = U
    return traj


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ---- Hyperparametres physiques (identiques au script d'origine) ----
    eps_goy, lmb = 0.5, 2
    a_coef = eps_goy / lmb
    b_coef = (eps_goy - 1) / lmb ** 2
    nu_coef = 1.0e-7

    # ---- Chargement des donnees : reel si disponible, sinon demo synthetique ----
    n_shells = 10          # reduit pour que la demo tourne vite (mettre 22 avec les vraies donnees)
    n_t = 400              # reduit de meme (90090 dans le fichier d'origine)

    try:
        full_data = np.loadtxt(r"/Odyssey/private/s26calme/code_stage/GOY-main/data.dat")
        data_spin_up = full_data[int(0.1 * full_data.shape[0]):, :]
        U = torch.from_numpy(data_spin_up).float()
        n_shells = U.shape[1] // 2
        n_t = U.shape[0]
        U_re, U_im = U[:, 0::2], U[:, 1::2]
        t_values = torch.linspace(0.0, 1.0, n_t)
        print(f"Donnees reelles chargees : n_shells={n_shells}, n_t={n_t}")
    except (FileNotFoundError, OSError):
        print("data.dat introuvable -> generation de donnees synthetiques pour la demo "
              f"(n_shells={n_shells}, n_t={n_t}).")
        traj = _simulate_goy_reference(n_shells, n_t, a_coef, b_coef, nu_coef, dt=2e-3, seed=0)
        U_re = torch.from_numpy(traj.real).float()
        U_im = torch.from_numpy(traj.imag).float()
        t_values = torch.linspace(0.0, (n_t - 1) * 2e-3, n_t)

    k_values = torch.arange(1, n_shells + 1, dtype=torch.float32)

    # shells "entierement observes" (a adapter -- ici les 4 du milieu, a titre d'exemple)
    mid = 6
    shells_to_observe = [mid - 1, mid, mid + 1]
    k_obs_idx = torch.tensor([s for s in shells_to_observe if 0 <= s < n_shells], dtype=torch.long)

    data = GOYData(k_values=k_values, t_values=t_values, U_re=U_re, U_im=U_im,
                   k_obs_idx=k_obs_idx, idx_t0=0)

    model = PINN(layers=[2, 32, 32, 2], w0=30.0, fourier_mapping_size=32, fourier_scale=10.0)

    cfg = ANaGRAMConfigGOY(
        epsilon=1e-5,
        eta_max=1.0,
        n_iter=1000,
        w_ic=1.0,
        w_obs=1.0,
        w_pde=1.0,
        n_t_colloc=200,     # x n_shells points PDE par iteration
        n_t_obs=200,        # x n_shells_obs points d'observation par iteration
        pde_start_iter=0,  # mettre >0 pour repartir "donnees seules" avant d'activer la physique
        seed=0,
    )

    model, history, idx_c_history, idx_o_history, history_obs, history_ic, history_phy = train_anagram_goy(  #ici
        model, data, a_coef, b_coef, nu_coef, cfg, device=device)

    out_dir = "/Odyssey/private/s26calme/code_stage/GOY-main/anagram"

    # -------------------------------------------------------------------
    # 7.0bis Verification des batches : shells observes + couverture temporelle
    # -------------------------------------------------------------------
    fig_snap = plot_batch_snapshot(data, cfg, device=device)
    fig_snap.savefig(f"{out_dir}/anagram_goy_batch_snapshot.png", dpi=140)

    fig_cov = plot_batch_time_coverage(idx_c_history, idx_o_history, t_values, idx_t0=data.idx_t0)
    fig_cov.savefig(f"{out_dir}/anagram_goy_batch_time_coverage.png", dpi=140)

    # -------------------------------------------------------------------
    # 7.1 Reconstruction des objets fonctionnels necessaires aux plots
    #     (theta final + u_pair_batch, comme dans l'entrainement)
    # -------------------------------------------------------------------
    ref_params = params_dict(model)
    theta_final = flatten_params(ref_params)
    O_k = 0.125 * 2.0 ** torch.arange(1, n_shells + 1, dtype=torch.float32)
    u_pair_batch, _, _ = make_functional_ops(model, ref_params, O_k, a_coef, b_coef, nu_coef)

    # -------------------------------------------------------------------
    # 7.2 Courbe de convergence (loss totale ponderee)
    # -------------------------------------------------------------------
    fig0, ax0 = plt.subplots(figsize=(7, 4.5))
    ax0.semilogy(history,label='Total Loss')
    ax0.semilogy(history_obs,label='Obs Loss')
    ax0.semilogy(history_ic,label='IC Loss')
    ax0.semilogy(history_phy,label='Physic Loss')
    ax0.legend()
    ax0.set_xlabel("iteration"); ax0.set_ylabel("loss totale ponderee")
    ax0.set_title("Convergence ANaGRAM"); ax0.grid(True, alpha=0.3)
    fig0.tight_layout()
    fig0.savefig(f"{out_dir}/anagram_goy_convergence.png", dpi=140)

    # -------------------------------------------------------------------
    # 7.3 Predictions vs verite sur TOUS les shells (Re et Im, grille complete)
    # -------------------------------------------------------------------
    fig1, u_re_pred_full, u_im_pred_full = plot_full_grid_predictions_anagram(
        u_pair_batch, theta_final, k_values, t_values, U_re, U_im,
        shells_to_plot=None,     # None = TOUS les shells
        chunk_size=2000,
    )
    fig1.savefig(f"{out_dir}/anagram_goy_full_grid_prediction.png", dpi=140)

    # -------------------------------------------------------------------
    # 7.4 RMSE du module au carre, par shell
    # -------------------------------------------------------------------
    fig2, rmse_per_shell = plot_rmse_module_per_shell_anagram(
        u_pair_batch, theta_final, k_values, t_values, U_re, U_im, chunk_size=2000,
    )
    fig2.savefig(f"{out_dir}/anagram_goy_rmse_module_per_shell.png", dpi=140)

    # -------------------------------------------------------------------
    # 7.5 Spectre energetique par shell (log(mean_t |U|^2))
    # -------------------------------------------------------------------
    fig3, log_E_pred, log_E_exa = plot_energy_spectrum_per_shell_anagram(
        u_pair_batch, theta_final, k_values, t_values, U_re, U_im, chunk_size=2000,
    )
    fig3.savefig(f"{out_dir}/anagram_goy_energy_spectrum_per_shell.png", dpi=140)

    # -------------------------------------------------------------------
    # 7.6 Kurtosis du module par shell (log(kurtosis(|U|)))
    # -------------------------------------------------------------------
    fig4, log_kurt_pred, log_kurt_exa = plot_kurtosis_per_shell_anagram(
        u_pair_batch, theta_final, k_values, t_values, U_re, U_im, chunk_size=2000,
    )
    fig4.savefig(f"{out_dir}/anagram_goy_kurtosis_per_shell.png", dpi=140)

    print("Figures sauvegardees :")
    print(f"  - {out_dir}/anagram_goy_batch_snapshot.png")
    print(f"  - {out_dir}/anagram_goy_batch_time_coverage.png")
    print(f"  - {out_dir}/anagram_goy_convergence.png")
    print(f"  - {out_dir}/anagram_goy_full_grid_prediction.png")
    print(f"  - {out_dir}/anagram_goy_rmse_module_per_shell.png")
    print(f"  - {out_dir}/anagram_goy_energy_spectrum_per_shell.png")
    print(f"  - {out_dir}/anagram_goy_kurtosis_per_shell.png")
