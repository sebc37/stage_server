import numpy as np

def dudt(U, t, n, K, eps, lmb, nu):
    """Time derivative for shell *n* in the GOY model.

    Parameters
    ----------
    U : array_like
        Current state vector for all shells.
    t : float
        Current time (unused in autonomous GOY, kept for API compatibility).
    n : int
        Index of the shell for which the derivative is requested.
    K : sequence
        Coefficients for each shell.
    eps : float
        Non‑linear coefficient in the GOY model.
    lmb : float
        Scale ratio between consecutive shells.
    nu : float
        Viscosity coefficient.

    Returns
    -------
    float
        Time derivative \(\dot U_n\).
    """
    return (
        K[n] *
        (U[n+1] * U[n+2]
         - (eps / lmb) * U[n-1] * U[n+1]
         + ((eps - 1) / lmb**2) * U[n-2] * U[n-1])
        - nu * (K[n]**2) * U[n]
    )


def RK4(U0, t0, n, t, dt, K, eps, lmb, nu):
    """Integrate the GOY shell system using classical RK4.

    The method advances the state from time ``t0`` to ``t`` with a fixed
    step ``dt``.  ``U0`` should be a one‑dimensional array of length ``n``
    containing the initial values for all shells.  Only shells with indices
    ``2 <= j < n-2`` are updated (consistent with the original implementation).

    Returns an array of shape ``(Nstep, len(U0))`` containing the state at
    each timestep (the initial state is _not_ included).
    """
    Nstep = int((t - t0) / dt)
    Un = U0.copy()
    U = np.zeros((Nstep, len(U0)))

    for i in range(Nstep):
        # allocate stage derivatives
        k1 = np.zeros_like(Un)
        k2 = np.zeros_like(Un)
        k3 = np.zeros_like(Un)
        k4 = np.zeros_like(Un)

        # compute k1 through k4 for each shell independently
        for j in range(2, n - 2):
            k1[j-2] = dudt(Un, t0, j, K, eps, lmb, nu)
        k1 *= dt

        for j in range(2, n - 2):
            k2[j-2] = dudt(Un + 0.5 * k1, t0 + 0.5 * dt, j, K, eps, lmb, nu)
        k2 *= dt

        for j in range(2, n - 2):
            k3[j-2] = dudt(Un + 0.5 * k2, t0 + 0.5 * dt, j, K, eps, lmb, nu)
        k3 *= dt

        for j in range(2, n - 2):
            k4[j-2] = dudt(Un + k3, t0 + dt, j, K, eps, lmb, nu)
        k4 *= dt

        Un = Un + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t0 += dt

        U[i] = Un
        
    return U
        