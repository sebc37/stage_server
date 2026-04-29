"""
goy.py  –  Python wrapper for the GOY shell model library (goy_lib.so)

Usage
-----
    from goy import GoyModel
    import numpy as np

    model = GoyModel()          # default parameters (N=22, same as original code)

    # build initial conditions however you like
    sh = model.shell_wavenumbers()
    Xpp = sh**(-1/3)
    Ypp = np.full(model.N, 1e-4)
    Xp  = Xpp.copy()
    Yp  = Ypp.copy()

    # integrate 10 000 steps
    (Xp_new, Yp_new), (X_new, Y_new) = model.integrate(Xpp, Ypp, Xp, Yp, n_steps=10_000)

    # the two returned states are:
    #   (Xp_new, Yp_new)  →  state at  t + (n_steps - 1) * dt
    #   (X_new,  Y_new)   →  state at  t +  n_steps      * dt
"""

import ctypes
import os
import numpy as np

# ── locate the .so next to this file ──────────────────────────────────────────
_LIB_NAME = "goy_lib.so"
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, _LIB_NAME)

# ── C struct mirroring GoyParams ──────────────────────────────────────────────
class _GoyParams(ctypes.Structure):
    _fields_ = [
        ("N",        ctypes.c_int),
        ("k0",       ctypes.c_double),
        ("lmb",      ctypes.c_double),
        ("eps",      ctypes.c_double),
        ("nu",       ctypes.c_double),
        ("dt",       ctypes.c_double),
        ("force",    ctypes.c_double),
        ("N_force",  ctypes.c_int),
        ("force_rnd",ctypes.c_int),
    ]


class GoyModel:
    """
    GOY shell-model integrator backed by a compiled C shared library.

    Parameters
    ----------
    N         : number of shells                     (default 22)
    k0        : largest-scale wave-number            (default 0.125)
    lmb       : ratio between consecutive shells     (default 2.0)
    eps       : NL coupling parameter                (default 0.5)
    nu        : viscosity                            (default 1e-7)
    dt        : time step                            (default 1e-5)
    force     : forcing amplitude                    (default 0.005)
    N_force   : index of the forced shell            (default 4)
    force_rnd : 1 = random forcing, 0 = deterministic (default 0)
    lib_path  : path to goy_lib.so                   (default: same directory as this file)
    """

    def __init__(self,
                 N=22,
                 k0=0.125,
                 lmb=2.0,
                 eps=0.5,
                 nu=1e-7,
                 dt=1e-5,
                 force=0.005,
                 N_force=4,
                 force_rnd=0,
                 lib_path=None):

        if N > 64:
            raise ValueError("N must be ≤ 64 (recompile goy_lib.c with larger N_MAX if needed)")

        self.N   = N
        self.dt  = dt
        self._lmb = lmb
        self._eps = eps
        self._nu  = nu

        # load library
        path = lib_path or _LIB_PATH
        self._lib = ctypes.CDLL(path)

        # goy_init(const GoyParams *)
        self._lib.goy_init.restype  = None
        self._lib.goy_init.argtypes = [ctypes.POINTER(_GoyParams)]

        # int goy_integrate(Xpp, Ypp, Xp, Yp, n_steps, out_Xp, out_Yp, out_X, out_Y)
        _dp = ctypes.POINTER(ctypes.c_double)
        self._lib.goy_integrate.restype  = ctypes.c_int
        self._lib.goy_integrate.argtypes = [
            _dp, _dp,           # Xpp, Ypp  (input)
            _dp, _dp,           # Xp,  Yp   (input)
            ctypes.c_int,       # n_steps
            _dp, _dp,           # out_Xp, out_Yp
            _dp, _dp,           # out_X,  out_Y
        ]

        # initialise C state
        params = _GoyParams(N=N, k0=k0, lmb=lmb, eps=eps, nu=nu,
                            dt=dt, force=force, N_force=N_force,
                            force_rnd=force_rnd)
        self._lib.goy_init(ctypes.byref(params))

        # cache shell wave-numbers for convenience
        self._sh = k0 * lmb ** np.arange(N,dtype=np.float64)

    # ── helpers ───────────────────────────────────────────────────────────────

    def shell_wavenumbers(self) -> np.ndarray:
        """Return the array of shell wave-numbers k_n = k0 * lmb^n."""
        return self._sh.copy()

    @staticmethod
    def _c_arr(a: np.ndarray):
        """Return a ctypes pointer to a contiguous float64 array."""
        a = np.ascontiguousarray(a, dtype=np.float64)
        return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), a

    # ── main API ──────────────────────────────────────────────────────────────

    def init_fields(self,Xpp=None,Ypp=None) -> tuple:
        """
        Compute the two-step initial condition (Xpp, Ypp, Xp, Yp) exactly as
        the original init_fields() does:
          - Xpp  = k_n^{-1/3},  Ypp = 1e-4
          - NXpp, NYpp computed from (Xpp, Ypp)
          - Xp   = A * (Xpp + dt * NXpp)   ← one Euler step
          - Yp   = A * (Ypp + dt * NYpp)

        Returns
        -------
        (Xpp, Ypp, Xp, Yp)  all as float64 numpy arrays of length N
        """
        import numpy as _np
        sh = self._sh
        
        if type(Xpp)==type(None) and type(Ypp)==type(None):
            Xpp = sh ** (-1.0 / 3.0)
            Ypp = _np.full(self.N, 1e-4,dtype=np.float64)
        

        # need A[] – recompute here (stored in C side, replicate in Python)
        A = _np.exp(-self._nu * sh**2 * self.dt)

        # call lib to get NX/NY for one step (hack: integrate 1 step from same state,
        # but we only need NXpp → compute manually in Python with same formulas)
        lmb = self._lmb; eps = self._eps; N = self.N
        A2 = _np.full(N, -eps/lmb)
        A3 = _np.full(N, -(1-eps)/(lmb**2))

        def _NX(ax, ay):
            res = _np.zeros(N,dtype=np.float64)
            res[0] = ax[2]*ay[1]+ay[2]*ax[1]
            res[1] = ax[3]*ay[2]+ay[3]*ax[2]+A2[1]*(ax[2]*ay[0]+ay[2]*ax[0])
            for k in range(2, N-2):
                res[k] = (ax[k+2]*ay[k+1]+ay[k+2]*ax[k+1]) \
                        + A2[k]*(ax[k+1]*ay[k-1]+ay[k+1]*ax[k-1]) \
                        + A3[k]*(ax[k-1]*ay[k-2]+ay[k-1]*ax[k-2])
            res[N-2] = A2[N-2]*(ax[N-1]*ay[N-3]+ay[N-1]*ax[N-3]) \
                     + A3[N-2]*(ax[N-3]*ay[N-4]+ay[N-3]*ax[N-4])
            res[N-1] = A3[N-1]*(ax[N-2]*ay[N-3]+ay[N-2]*ax[N-3])
            return res * sh

        def _NY(ax, ay):
            res = _np.zeros(N,dtype=np.float64)
            res[0] = ax[2]*ax[1]-ay[1]*ay[2]
            res[1] = ax[3]*ax[2]-ay[2]*ay[3]+A2[1]*(ax[2]*ax[0]-ay[2]*ay[0])
            for k in range(2, N-2):
                res[k] = ax[k+2]*ax[k+1]-ay[k+1]*ay[k+2] \
                        + A2[k]*(ax[k+1]*ax[k-1]-ay[k+1]*ay[k-1]) \
                        + A3[k]*(ax[k-1]*ax[k-2]-ay[k-1]*ay[k-2])
            res[N-2] = A2[N-2]*(ax[N-1]*ax[N-3]+ay[N-1]*ay[N-3]) \
                     + A3[N-2]*(ax[N-3]*ax[N-4]+ay[N-3]*ay[N-4])
            res[N-1] = A3[N-1]*(ax[N-2]*ax[N-3]-ay[N-2]*ay[N-3])
            return res * sh

        NXpp = _NX(Xpp, Ypp)
        NYpp = _NY(Xpp, Ypp)
        Xp = A * (Xpp + self.dt * NXpp)
        Yp = A * (Ypp + self.dt * NYpp)
        # print(f"Xpp0 : {Xpp}")
        # print(f"Ypp0 : {Ypp}")
        # print(f"Xp : {Xp}")
        # print(f"Yp : {Yp}")
        return Xpp, Ypp, Xp, Yp

    def integrate(self,
                  Xpp: np.ndarray, Ypp: np.ndarray,
                  Xp:  np.ndarray, Yp:  np.ndarray,
                  n_steps: int):
        """
        Integrate the GOY model for *n_steps* Adams-Bashforth steps.

        Parameters
        ----------
        Xpp, Ypp : real/imaginary parts of the complex field at  t - dt
        Xp,  Yp  : real/imaginary parts of the complex field at  t
        n_steps  : number of integration steps

        Returns
        -------
        (Xp_out, Yp_out) : state at  t + (n_steps - 1) * dt
        (X_out,  Y_out)  : state at  t +  n_steps      * dt

        Raises
        ------
        RuntimeError if a NaN is detected during integration.
        """
        N = self.N
        for arr in (Xpp, Ypp, Xp, Yp):
            if len(arr) != N:
                raise ValueError(f"All input arrays must have length N={N}")

        out_Xp = np.zeros(N, dtype=np.float64)
        out_Yp = np.zeros(N, dtype=np.float64)
        out_X  = np.zeros(N, dtype=np.float64)
        out_Y  = np.zeros(N, dtype=np.float64)

        p_Xpp, _Xpp = self._c_arr(Xpp)
        p_Ypp, _Ypp = self._c_arr(Ypp)
        p_Xp,  _Xp  = self._c_arr(Xp)
        p_Yp,  _Yp  = self._c_arr(Yp)
        p_oXp  = out_Xp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_oYp  = out_Yp.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_oX   = out_X .ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        p_oY   = out_Y .ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        ret = self._lib.goy_integrate(
            p_Xpp, p_Ypp, p_Xp, p_Yp,
            ctypes.c_int(n_steps),
            p_oXp, p_oYp, p_oX, p_oY
        )

        if ret != 0:
            raise RuntimeError("NaN detected during GOY integration")

        return (out_Xp, out_Yp), (out_X, out_Y)


# ── quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    model = GoyModel()
    sh = model.shell_wavenumbers()

    Xpp = sh ** (-1.0 / 3.0)
    Ypp = np.full(model.N, 1e-4)
    Xp, Yp = Xpp.copy(), Ypp.copy()

    print(f"Integrating {model.N} shells for 100 000 steps …")
    t0 = time.perf_counter()
    (Xp2, Yp2), (X2, Y2) = model.integrate(Xpp, Ypp, Xp, Yp, n_steps=100_000)
    elapsed = time.perf_counter() - t0

    print(f"Done in {elapsed:.3f} s")
    print(f"X[0]  = {X2[0]:.6e}")
    print(f"Xp[0] = {Xp2[0]:.6e}")
    energy = 0.5 * np.sum(X2**2 + Y2**2)
    print(f"Total energy = {energy:.6e}")
