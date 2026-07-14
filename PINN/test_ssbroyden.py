"""
ss_broyden.py
=============
A general-purpose SS-Broyden (Structured Symmetric Broyden) quasi-Newton
optimizer for Physics-Informed Neural Networks (PINNs) in PyTorch.

Usage
-----
    optimizer = SSBroydenOptimizer(model, loss_fn, **kwargs)
    for outer in range(n_outer):
        loss = optimizer.step(*data_args)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple, List
import time
import warnings
import os
#PATH = r"/Odyssey/private/s26calme/code_stage/PINN/"

# ============================================================
# 1. LINE SEARCH
# ============================================================

def _flat_grad(
    loss_fn : Callable,
    theta   : torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient of loss_fn(theta) w.r.t. theta.
    Returns a detached flat tensor.
    
    loss_fn must be a function  f(theta) -> scalar tensor
    where theta is directly used in the computation graph.
    """
    theta = theta.detach().clone().requires_grad_(True)
    loss  = loss_fn(theta)

    loss.backward()

    if theta.grad is None:
        raise RuntimeError(
            "_flat_grad: theta.grad is None.\n"
            "The computation graph is broken between theta and the loss.\n"
            "Make sure _theta_loss uses torch.func.functional_call."
        )

    grad = theta.grad.detach().clone()
    return grad



def _armijo_wolfe_ls(
    loss_fn    : Callable,
    theta      : torch.Tensor,
    grad       : torch.Tensor,
    direction  : torch.Tensor,
    c1         : float = 1e-4,
    c2         : float = 0.9,
    alpha_init : float = 1.0,
    max_iter   : int   = 25,
) -> Tuple[float, torch.Tensor, torch.Tensor, float]:
    """
    Strong Wolfe line search via bisection.

    Returns
    -------
    alpha      : accepted step size
    theta_new  : updated parameters
    grad_new   : gradient at theta_new
    loss_new   : loss at theta_new
    """
    f0    = loss_fn(theta).item()
    slope = (grad @ direction).item()

    # If direction is not a descent direction, fall back to -grad
    if slope >= 0.0:
        warnings.warn("LSSearch: direction not descent, using -grad.")
        direction = -grad
        slope     = (grad @ direction).item()

    alpha    = alpha_init
    alpha_lo = 0.0
    alpha_hi = float("inf")

    theta_new = theta
    grad_new  = grad
    loss_new  = f0

    for _ in range(max_iter):
        theta_new = theta + alpha * direction
        loss_new  = loss_fn(theta_new).item()

        # Armijo condition (sufficient decrease)
        if loss_new > f0 + c1 * alpha * slope:
            alpha_hi = alpha
            alpha    = 0.5 * (alpha_lo + alpha_hi)
            continue

        grad_new  = _flat_grad(loss_fn, theta_new)
        slope_new = (grad_new @ direction).item()

        # Strong Wolfe curvature condition
        if abs(slope_new) <= c2 * abs(slope):
            return alpha, theta_new, grad_new, loss_new

        if slope_new * (alpha_hi - alpha_lo) >= 0.0:
            alpha_hi = alpha_lo

        alpha_lo = alpha
        alpha    = (
            2.0 * alpha_lo
            if alpha_hi == float("inf")
            else 0.5 * (alpha_lo + alpha_hi)
        )

    # Did not fully converge — return best found
    theta_new = theta + alpha * direction
    loss_new  = loss_fn(theta_new).item()
    grad_new  = _flat_grad(loss_fn, theta_new)
    return alpha, theta_new, grad_new, loss_new


# ============================================================
# 2. INVERSE-HESSIAN UPDATE RULES
# ============================================================

def _ssbroyden2_update(
    H       : torch.Tensor,
    s       : torch.Tensor,
    y       : torch.Tensor,
    damping : float = 1e-8,
) -> torch.Tensor:
    """
    SS-Broyden-2 symmetric rank-1 update of the inverse Hessian.

    Secant condition:  H_new @ y ≈ s
    Symmetry enforced: H_new = (H_new + H_new^T) / 2

    Parameters
    ----------
    H       : (n, n) current inverse Hessian estimate
    s       : (n,)   step vector        s = theta_new - theta_old
    y       : (n,)   gradient diff      y = grad_new  - grad_old
    damping : skip update if curvature is too small

    Returns
    -------
    H_new : (n, n) updated inverse Hessian
    """
    n  = H.shape[0]
    sy = (s @ y).item()

    if abs(sy) < damping:
        return H                          # skip — curvature too flat

    Hy = H @ y                            # (n,)
    u  = s - Hy                           # residual of secant condition
    uy = (u @ y).item()

    if abs(uy) < damping:
        return H                          # skip — degenerate update

    # Symmetric rank-1 update
    H_new = H + torch.outer(u, u) / uy

    # Enforce exact symmetry
    H_new = 0.5 * (H_new + H_new.T)

    # Safety check
    if not torch.isfinite(H_new).all():
        warnings.warn("SSBroyden2: non-finite H after update → reset to I.")
        return torch.eye(n, dtype=H.dtype, device=H.device)

    return H_new


def _bfgs_update(
    H       : torch.Tensor,
    s       : torch.Tensor,
    y       : torch.Tensor,
    damping : float = 1e-8,
) -> torch.Tensor:
    """
    Classic BFGS inverse Hessian update.
    Used as fallback or alternative to SSBroyden2.

    Parameters
    ----------
    H       : (n, n) current inverse Hessian estimate
    s       : (n,)   step vector
    y       : (n,)   gradient difference
    damping : curvature threshold

    Returns
    -------
    H_new : (n, n) updated inverse Hessian
    """
    n   = H.shape[0]
    sy  = (s @ y).item()

    if sy <= damping:
        return H                          # curvature condition not satisfied

    rho   = 1.0 / sy
    I     = torch.eye(n, dtype=H.dtype, device=H.device)
    A     = I - rho * torch.outer(s, y)
    B     = I - rho * torch.outer(y, s)
    H_new = A @ H @ B + rho * torch.outer(s, s)

    # Enforce symmetry
    H_new = 0.5 * (H_new + H_new.T)

    if not torch.isfinite(H_new).all():
        warnings.warn("BFGS: non-finite H after update → reset to I.")
        return torch.eye(n, dtype=H.dtype, device=H.device)

    return H_new


def _sr1_update(
    H       : torch.Tensor,
    s       : torch.Tensor,
    y       : torch.Tensor,
    damping : float = 1e-8,
    r       : float = 1e-8,
) -> torch.Tensor:
    """
    Symmetric Rank-1 (SR1) inverse Hessian update.
    More aggressive than BFGS, can capture indefinite curvature.

    Parameters
    ----------
    H       : (n, n) current estimate
    s       : (n,)   step
    y       : (n,)   gradient difference
    damping : minimum denominator threshold
    r       : skip-update safety ratio  (Nocedal & Wright §6.2)

    Returns
    -------
    H_new : (n, n) updated inverse Hessian
    """
    n   = H.shape[0]
    u   = s - H @ y
    uy  = (u @ y).item()

    # SR1 skip condition
    if abs(uy) < r * u.norm().item() * y.norm().item() + damping:
        return H

    H_new = H + torch.outer(u, u) / uy
    H_new = 0.5 * (H_new + H_new.T)

    if not torch.isfinite(H_new).all():
        warnings.warn("SR1: non-finite H → reset to I.")
        return torch.eye(n, dtype=H.dtype, device=H.device)

    return H_new


# ============================================================
# 3. PARAMETER FLATTENER
# ============================================================

class ParameterFlattener:
    """
    Bijective map between an nn.Module's trainable parameters
    and a flat 1-D torch.Tensor.

    Works with any nn.Module (MLP, PINN, DeepONet, …).
    """

    def __init__(self, model: nn.Module):
        self.model  = model
        self.names  : List[str]             = []
        self.shapes : List[torch.Size]      = []
        self.sizes  : List[int]             = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.names.append(name)
                self.shapes.append(param.shape)
                self.sizes.append(param.numel())

        self.total = sum(self.sizes)

    # ----------------------------------------------------------------
    def flatten(self) -> torch.Tensor:
        """Return a flat copy of all trainable parameters."""
        parts  = []
        params = dict(self.model.named_parameters())
        for name in self.names:
            parts.append(params[name].detach().view(-1))
        return torch.cat(parts)

    # ----------------------------------------------------------------
    def unflatten(self, theta: torch.Tensor) -> None:
        """Write flat vector theta back into model parameters (in-place)."""
        params = dict(self.model.named_parameters())
        idx    = 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            chunk = theta[idx : idx + size].view(shape)
            params[name].data.copy_(chunk)
            idx  += size

    # ----------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"ParameterFlattener("
            f"n_tensors={len(self.names)}, "
            f"total_dof={self.total})"
        )


# ============================================================
# 4. MAIN OPTIMIZER CLASS
# ============================================================

class SSBroydenOptimizer:
    """
    General-purpose SS-Broyden quasi-Newton optimizer for PINNs (PyTorch).

    Parameters
    ----------
    model : nn.Module
        Any trainable PyTorch model.
    loss_fn : Callable
        Scalar loss.  Signature:  loss = loss_fn(model, *args)
        *args are arbitrary data tensors (collocation pts, BCs, …).
    update_method : str
        One of  "ssbroyden2" | "bfgs" | "sr1".
        Default: "ssbroyden2".
    maxiter_inner : int
        Maximum quasi-Newton inner iterations per outer call.
    gtol : float
        Gradient-norm stopping tolerance for inner loop.
    initial_scale : bool
        Scale H with Barzilai–Borwein estimate on the very first step.
    ls_c1 : float
        Armijo sufficient-decrease constant.
    ls_c2 : float
        Wolfe curvature constant.
    ls_maxiter : int
        Max bisection steps in line search.
    damping : float
        Minimum denominator for secant updates.
    dtype : torch.dtype
        Floating-point precision for internal tensors.
        Recommended: torch.float64 for quasi-Newton.
    device : str | torch.device | None
        Device for internal tensors (defaults to model's device).
    verbose : bool
        Print inner-loop diagnostics.

    Examples
    --------
    Basic usage
    ~~~~~~~~~~~
    >>> def pinn_loss(model, xy_f, xy_bc):
    ...     # physics residual + boundary loss
    ...     ...
    ...     return loss

    >>> model    = MyPINN().double()
    >>> optimizer = SSBroydenOptimizer(model, pinn_loss)

    >>> for outer in range(500):
    ...     loss = optimizer.step(xy_f, xy_bc)
    ...     print(f"[{outer:03d}] loss = {loss:.3e}")

    Adam warm-up then SS-Broyden
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    >>> adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> for _ in range(1000):
    ...     adam.zero_grad()
    ...     loss_fn(model, xy_f, xy_bc).backward()
    ...     adam.step()

    >>> optimizer = SSBroydenOptimizer(model, loss_fn, maxiter_inner=30)
    >>> for outer in range(500):
    ...     loss = optimizer.step(xy_f, xy_bc)

    Save / restore
    ~~~~~~~~~~~~~~
    >>> optimizer.save_state("checkpoint.pt")
    >>> optimizer.load_state("checkpoint.pt")
    """

    # Valid update methods
    _METHODS = {"ssbroyden2", "bfgs", "sr1"}

    def __init__(
        self,
        model         : nn.Module,
        loss_fn       : Callable,
        update_method : str         = "ssbroyden2",
        maxiter_inner : int         = 30,
        gtol          : float       = 1e-9,
        initial_scale : bool        = True,
        ls_c1         : float       = 1e-4,
        ls_c2         : float       = 0.9,
        ls_maxiter    : int         = 25,
        damping       : float       = 1e-8,
        dtype         : torch.dtype = torch.float64,
        device        : Any         = None,
        verbose       : bool        = False,
    ):
        if update_method not in self._METHODS:
            raise ValueError(
                f"update_method must be one of {self._METHODS}, "
                f"got '{update_method}'."
            )

        self.model         = model
        self._loss_fn_user = loss_fn
        self.update_method = update_method
        self.maxiter_inner = maxiter_inner
        self.gtol          = gtol
        self.initial_scale = initial_scale
        self.ls_c1         = ls_c1
        self.ls_c2         = ls_c2
        self.ls_maxiter    = ls_maxiter
        self.damping       = damping
        self.dtype         = dtype
        self.verbose       = verbose

        # ---- device ----
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        # ---- flattener ----
        self.flattener = ParameterFlattener(model)
        n              = self.flattener.total

        print(f"SSBroydenOptimizer | method={update_method} | dof={n:,}")

        # ---- inverse Hessian (n × n, double precision) ----
        self.H = torch.eye(n, dtype=self.dtype, device=self.device)

        # ---- state flags ----
        self._first_step  = True
        self.n_outer_done = 0

        # ---- history ----
        self.history: Dict[str, List] = {
            "loss"     : [],
            "grad_norm": [],
            "n_inner"  : [],
            "alpha"    : [],
            "time_s"   : [],
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _theta_loss(self, args: tuple) -> Callable:
        """
        Return  f(theta) = loss_fn(model_with_theta, *args)
    
        Uses torch.func.functional_call so that theta stays in the
        autograd graph → gradients are correctly computed.
        
        Parameters
        ----------
        args : tuple
            Data arguments forwarded to the user loss_fn.
        
        Returns
        -------
        f : Callable
            f(theta) -> scalar tensor, differentiable w.r.t. theta.
        """
        names  = self.flattener.names
        shapes = self.flattener.shapes
        sizes  = self.flattener.sizes
        model  = self.model
        loss_fn_user = self._loss_fn_user

        def f(theta: torch.Tensor) -> torch.Tensor:
            # ---- Build parameter dict from flat theta ----
            param_dict = {}
            idx = 0
            for name, shape, size in zip(names, shapes, sizes):
                param_dict[name] = theta[idx : idx + size].reshape(shape)
                idx += size

            # ---- Functional model: no in-place .data write ----
            # We create a wrapper so that loss_fn_user(model, *args)
            # internally uses our differentiable param_dict
            class _FunctionalModel:
                """
                Thin wrapper: behaves like model but uses
                parameters from theta (differentiable).
                """
                def __call__(self_, xy):
                    return torch.func.functional_call(
                        model, param_dict, (xy,)
                    )

            functional_model = _FunctionalModel()
            return loss_fn_user(functional_model, *args)

        return f

    def _bb_scale(self, g: torch.Tensor) -> None:
        """Barzilai–Borwein initial scaling of H."""
        gg = (g @ g).item()
        if gg > 1e-30:
            scale  = 1.0 / (gg ** 0.5)
            n      = self.H.shape[0]
            self.H = scale * torch.eye(
                n, dtype=self.dtype, device=self.device
            )

    def _update_H(
        self,
        s: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """Dispatch to the chosen update rule."""
        if self.update_method == "ssbroyden2":
            self.H = _ssbroyden2_update(self.H, s, y, self.damping)
        elif self.update_method == "bfgs":
            self.H = _bfgs_update(self.H, s, y, self.damping)
        elif self.update_method == "sr1":
            self.H = _sr1_update(self.H, s, y, self.damping)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, *args) -> float:
        """
        Run one outer SS-Broyden macro-step.
        """
        t0    = time.time()
        f     = self._theta_loss(args)
        theta = self.flattener.flatten().to(self.device).to(self.dtype)

        # ---- Initial loss and gradient ----
        g        = _flat_grad(f, theta)
        loss_val = f(theta.detach().clone()).item()

        # ---- Barzilai–Borwein initial scaling ----
        if self._first_step and self.initial_scale:
            self._bb_scale(g)
            self._first_step = False

        n_inner    = 0
        last_alpha = 1.0

        for k in range(self.maxiter_inner):
            g_norm = g.norm().item()

            if self.verbose:
                print(
                    f"  [outer={self.n_outer_done:03d} "
                    f"inner={k:02d}] "
                    f"loss={loss_val:.4e}  "
                    f"‖g‖={g_norm:.3e}"
                )

            if g_norm < self.gtol:
                if self.verbose:
                    print(f"  → converged (‖g‖={g_norm:.2e} < gtol={self.gtol:.1e})")
                break

            # ---- Quasi-Newton direction ----
            p = -(self.H @ g)

            # ---- Line search ----
            alpha, theta_new, g_new, loss_new = _armijo_wolfe_ls(
                loss_fn    = f,
                theta      = theta,
                grad       = g,
                direction  = p,
                c1         = self.ls_c1,
                c2         = self.ls_c2,
                alpha_init = 1.0,
                max_iter   = self.ls_maxiter,
            )

            # ---- Secant pair ----
            s = (theta_new - theta).detach()
            y = (g_new     - g    ).detach()

            # ---- Inverse-Hessian update ----
            self._update_H(s, y)

            theta      = theta_new.detach()
            g          = g_new.detach()
            loss_val   = loss_new
            last_alpha = alpha
            n_inner   += 1

        # ---- Write final parameters back into model ----
        self.flattener.unflatten(theta.detach())

        # ---- History ----
        elapsed = time.time() - t0
        self.history["loss"].append(loss_val)
        self.history["grad_norm"].append(g.norm().item())
        self.history["n_inner"].append(n_inner)
        self.history["alpha"].append(last_alpha)
        self.history["time_s"].append(elapsed)
        self.n_outer_done += 1

        return loss_val

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_state(self, path: str) -> None:
        """
        Save the optimizer state (H, flat parameters, history) to a .pt file.

        Parameters
        ----------
        path : str
            File path, e.g.  "./checkpoints/ssbroyden.pt"
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "H"            : self.H.cpu(),
            "theta"        : self.flattener.flatten().cpu(),
            "n_outer_done" : self.n_outer_done,
            "history"      : self.history,
            "update_method": self.update_method,
        }
        torch.save(state, path)
        print(f"✓ SSBroyden state saved  →  {path}")

    def load_state(self, path: str) -> None:
        """
        Restore optimizer state from a .pt file.

        Parameters
        ----------
        path : str
            File path previously created with save_state().
        """
        state = torch.load(path, map_location=self.device)

        self.H             = state["H"].to(self.dtype).to(self.device)
        self.n_outer_done  = state["n_outer_done"]
        self.history       = state["history"]
        self._first_step   = False

        self.flattener.unflatten(
            state["theta"].to(self.dtype).to(self.device)
        )
        print(
            f"✓ SSBroyden state loaded ←  {path}  "
            f"(outer steps done: {self.n_outer_done})"
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_hessian(self) -> None:
        """Reset inverse Hessian to the identity matrix."""
        n      = self.H.shape[0]
        self.H = torch.eye(n, dtype=self.dtype, device=self.device)
        self._first_step = True
        print("✓ Inverse Hessian reset to I.")

    def print_summary(self) -> None:
        """Print a short summary of the optimisation history."""
        h = self.history
        if not h["loss"]:
            print("No steps recorded yet.")
            return

        print("=" * 60)
        print(f"SSBroydenOptimizer  |  method = {self.update_method}")
        print(f"Outer steps done    : {self.n_outer_done}")
        print(f"Initial loss        : {h['loss'][0]:.4e}")
        print(f"Final   loss        : {h['loss'][-1]:.4e}")
        print(f"Final   ‖grad‖      : {h['grad_norm'][-1]:.4e}")
        print(f"Total wall time     : {sum(h['time_s']):.1f} s")
        print(f"Avg inner iters     : {np.mean(h['n_inner']):.1f}")
        print("=" * 60)

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot loss and gradient-norm history.

        Parameters
        ----------
        save_path : str | None
            If given, save figure to this path instead of showing it.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("matplotlib not found — cannot plot history.")
            return

        h    = self.history
        iters = range(1, len(h["loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].semilogy(iters, h["loss"], "b-", lw=1.5)
        axes[0].set_xlabel("Outer iteration")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss history")
        axes[0].grid(True, which="both", alpha=0.4)

        axes[1].semilogy(iters, h["grad_norm"], "r-", lw=1.5)
        axes[1].set_xlabel("Outer iteration")
        axes[1].set_ylabel(r"$\|\nabla \mathcal{L}\|$")
        axes[1].set_title("Gradient norm history")
        axes[1].grid(True, which="both", alpha=0.4)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ History figure saved → {save_path}")
            plt.close()
        else:
            plt.show()

    def __repr__(self) -> str:
        return (
            f"SSBroydenOptimizer(\n"
            f"  method        = {self.update_method}\n"
            f"  dof           = {self.flattener.total:,}\n"
            f"  maxiter_inner = {self.maxiter_inner}\n"
            f"  gtol          = {self.gtol:.1e}\n"
            f"  ls_c1/c2      = {self.ls_c1}/{self.ls_c2}\n"
            f"  outer done    = {self.n_outer_done}\n"
            f")"
        )


# ============================================================
# 5. ADAM WARM-UP HELPER
# ============================================================

def adam_warmup(
    model      : nn.Module,
    loss_fn    : Callable,
    args       : tuple,
    n_steps    : int   = 1000,
    lr         : float = 1e-3,
    print_every: int   = 100,
) -> List[float]:
    """
    Run a standard Adam warm-up before switching to SS-Broyden.

    Parameters
    ----------
    model       : nn.Module  (will be modified in-place)
    loss_fn     : Callable   loss_fn(model, *args) -> scalar tensor
    args        : tuple      data arguments forwarded to loss_fn
    n_steps     : int        number of Adam iterations
    lr          : float      Adam learning rate
    print_every : int        print frequency

    Returns
    -------
    losses : List[float]   loss at each step
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses    = []

    print(f"Adam warm-up: {n_steps} steps, lr={lr:.1e}")
    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        loss = loss_fn(model, *args)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % print_every == 0:
            print(f"  Adam [{step:6d}/{n_steps}]  loss = {loss.item():.4e}")

    print(f"Adam warm-up done | final loss = {losses[-1]:.4e}")
    return losses


# ============================================================
# 6. COMPLETE PINN EXAMPLE  (2-D Diffusion–Reaction)
# ============================================================

# if __name__ == "__main__":

#     import torch.nn.functional as F

#     # ---- reproducibility ----
#     torch.manual_seed(1234)
#     torch.set_default_dtype(torch.float64)

#     DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     K_REACT  = 0.1
#     PI       = torch.tensor(torch.pi, dtype=torch.float64, device=DEVICE)

#     # ------------------------------------------------------------------
#     # 6-a.  Periodic Fourier embedding
#     # ------------------------------------------------------------------
#     def periodic_embed(xy: torch.Tensor, freqs=(1, 3, 6)) -> torch.Tensor:
#         """
#         Map (x, y) ∈ ℝ² → Fourier features ∈ ℝ^{4·len(freqs)}.

#         Parameters
#         ----------
#         xy    : (batch, 2) tensor
#         freqs : tuple of integer frequencies

#         Returns
#         -------
#         feats : (batch, 4*len(freqs)) tensor
#         """
#         x, y  = xy[:, 0:1], xy[:, 1:2]
#         parts = []
#         for k in freqs:
#             parts += [
#                 torch.sin(k * PI * x),
#                 torch.cos(k * PI * x),
#                 torch.sin(k * PI * y),
#                 torch.cos(k * PI * y),
#             ]
#         return torch.cat(parts, dim=1)

#     # ------------------------------------------------------------------
#     # 6-b.  PINN architecture
#     # ------------------------------------------------------------------
#     class ReactionPINN(nn.Module):
#         """
#         Fully-connected PINN with periodic Fourier embedding.

#         Architecture:
#             Fourier embed (12) → [Linear → Tanh] × 7 → Linear (1)
#         """

#         def __init__(self, units: int = 44, freqs=(1, 3, 6)):
#             super().__init__()
#             self.freqs   = freqs
#             in_dim       = 4 * len(freqs)           # 12

#             self.net = nn.Sequential(
#                 nn.Linear(in_dim, units), nn.Tanh(),
#                 nn.Linear(units,  units), nn.Tanh(),
#                 nn.Linear(units,  units), nn.Tanh(),
#                 nn.Linear(units,  units), nn.Tanh(),
#                 nn.Linear(units,  units), nn.Tanh(),
#                 nn.Linear(units,  units), nn.Tanh(),
#                 nn.Linear(units,  units), nn.Tanh(),
#                 nn.Linear(units,  1),
#             )
#             self._init_weights()

#         def _init_weights(self):
#             for layer in self.net:
#                 if isinstance(layer, nn.Linear):
#                     nn.init.xavier_normal_(layer.weight)
#                     nn.init.zeros_(layer.bias)

#         def forward(self, xy: torch.Tensor) -> torch.Tensor:
#             """
#             Parameters
#             ----------
#             xy : (batch, 2) tensor

#             Returns
#             -------
#             u  : (batch, 1) tensor
#             """
#             feat = periodic_embed(xy, self.freqs)
#             return self.net(feat)

#     # ------------------------------------------------------------------
#     # 6-c.  Exact solution and source term
#     # ------------------------------------------------------------------
#     def u_exact(xy: torch.Tensor) -> torch.Tensor:
#         x, y = xy[:, 0], xy[:, 1]
#         return torch.sin(3 * PI * x) * torch.cos(3 * PI * y)

#     def f_source(xy: torch.Tensor) -> torch.Tensor:
#         u   = u_exact(xy)
#         lap = -18.0 * PI**2 * u
#         return lap - K_REACT * u**2

#     # ------------------------------------------------------------------
#     # 6-d.  Physics residual and loss
#     # ------------------------------------------------------------------
#     def physics_residual(
#         model : nn.Module,
#         xy    : torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Residual of  Δu − k·u² = f  evaluated on collocation points.

#         Uses torch.autograd.grad for u_xx and u_yy.
#         """
#         xy    = xy.requires_grad_(True)
#         u     = model(xy)                        # (N, 1)

#         # First derivatives
#         grads = torch.autograd.grad(
#             u, xy,
#             grad_outputs = torch.ones_like(u),
#             create_graph = True,
#         )[0]                                     # (N, 2)
#         u_x, u_y = grads[:, 0:1], grads[:, 1:2]

#         # Second derivatives
#         u_xx = torch.autograd.grad(
#             u_x, xy,
#             grad_outputs = torch.ones_like(u_x),
#             create_graph = True,
#         )[0][:, 0:1]

#         u_yy = torch.autograd.grad(
#             u_y, xy,
#             grad_outputs = torch.ones_like(u_y),
#             create_graph = True,
#         )[0][:, 1:2]

#         lap = u_xx + u_yy
#         f   = f_source(xy.detach())[:, None]
#         return lap - K_REACT * u**2 - f

#     def pinn_loss(
#         model  : nn.Module,
#         xy_col : torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Mean-squared PDE residual (no boundary loss needed
#         because the solution is periodic and the embedding
#         already encodes periodicity).
#         """
#         R = physics_residual(model, xy_col)
#         return torch.mean(R**2)

#     # ------------------------------------------------------------------
#     # 6-e.  Collocation points (Latin Hypercube)
#     # ------------------------------------------------------------------
#     try:
#         from pyDOE import lhs as lhs_np
#         xy_np = -1.0 + 2.0 * lhs_np(2, 10_000)
#     except ImportError:
#         xy_np = -1.0 + 2.0 * np.random.rand(10_000, 2)

#     xy_col = torch.tensor(xy_np, dtype=torch.float64, device=DEVICE)

#     # ------------------------------------------------------------------
#     # 6-f.  Model
#     # ------------------------------------------------------------------
#     model = ReactionPINN(units=44).to(DEVICE).double()
#     print(model)
#     n_params = sum(p.numel() for p in model.parameters())
#     print(f"Total trainable parameters: {n_params:,}")

#     # ------------------------------------------------------------------
#     # 6-g.  Adam warm-up
#     # ------------------------------------------------------------------
#     adam_losses = adam_warmup(
#         model       = model,
#         loss_fn     = pinn_loss,
#         args        = (xy_col,),
#         n_steps     = 1000,
#         lr          = 1e-3,
#         print_every = 200,
#     )

#     # ------------------------------------------------------------------
#     # 6-h.  SS-Broyden refinement
#     # ------------------------------------------------------------------
#     optimizer = SSBroydenOptimizer(
#         model         = model,
#         loss_fn       = pinn_loss,
#         update_method = "ssbroyden2",
#         maxiter_inner = 30,
#         gtol          = 1e-9,
#         initial_scale = True,
#         ls_c1         = 1e-4,
#         ls_c2         = 0.9,
#         ls_maxiter    = 25,
#         damping       = 1e-8,
#         verbose       = False,
#     )

#     print("\nSS-Broyden refinement...")
#     t0 = time.time()
#     for outer in range(200):
#         loss = optimizer.step(xy_col)
#         if (outer + 1) % 10 == 0:
#             print(f"  [{outer+1:03d}] loss = {loss:.4e}")

#     print(f"\nFinished in {time.time()-t0:.1f} s")
#     optimizer.print_summary()

#     # ------------------------------------------------------------------
#     # 6-i.  Evaluation
#     # ------------------------------------------------------------------
#     Nx = Ny = 100
#     x  = np.linspace(-1, 1, Nx)
#     y  = np.linspace(-1, 1, Ny)
#     XX, YY = np.meshgrid(x, y)

#     xy_test = torch.tensor(
#         np.stack([XX.ravel(), YY.ravel()], axis=1),
#         dtype  = torch.float64,
#         device = DEVICE,
#     )

#     model.eval()
#     with torch.no_grad():
#         u_pred = model(xy_test).cpu().numpy().ravel()

#     u_true = (
#         np.sin(3 * np.pi * XX.ravel()) *
#         np.cos(3 * np.pi * YY.ravel())
#     )

#     rel_l2 = (
#         np.linalg.norm(u_pred - u_true) /
#         np.linalg.norm(u_true)
#     )
#     linf = np.max(np.abs(u_pred - u_true))

#     print("=" * 60)
#     print(f"Relative L2 error : {rel_l2:.3e}  ({rel_l2*100:.4f} %)")
#     print(f"L∞ error          : {linf:.3e}")
#     print("=" * 60)

#     # ------------------------------------------------------------------
#     # 6-j.  Save
#     # ------------------------------------------------------------------
#     os.makedirs(PATH +"checkpoints", exist_ok=True)
#     optimizer.save_state(PATH + "checkpoints/ssbroyden_state.pt")
#     torch.save(model.state_dict(), PATH +"checkpoints/pinn_model.pt")
#     print("Model saved.")

#     # ------------------------------------------------------------------
#     # 6-k.  Plot
#     # ------------------------------------------------------------------
#     try:
#         import matplotlib.pyplot as plt

#         U_pred = u_pred.reshape(Ny, Nx)
#         U_true = u_true.reshape(Ny, Nx)
#         U_err  = np.abs(U_pred - U_true)

#         fig, axes = plt.subplots(1, 3, figsize=(15, 4))
#         titles = ["Exact", "PINN (SS-Broyden)", "Absolute error"]
#         fields = [U_true, U_pred, U_err]
#         cmaps  = ["viridis", "viridis", "inferno"]

#         for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
#             im = ax.imshow(
#                 field,
#                 origin="lower",
#                 extent=[-1, 1, -1, 1],
#                 cmap=cmap,
#                 aspect="auto",
#             )
#             plt.colorbar(im, ax=ax, fraction=0.046)
#             ax.set_title(title)
#             ax.set_xlabel("x")
#             ax.set_ylabel("y")

#         plt.tight_layout()
#         plt.savefig(PATH + "checkpoints/ssbroyden_result.png", dpi=300)
#         plt.close()
#         print("✓ Figure saved → ./checkpoints/ssbroyden_result.png")

#         optimizer.plot_history(
#             save_path=PATH + "checkpoints/ssbroyden_history.png"
#         )

#     except ImportError:
#         print("matplotlib not available — skipping plots.")
