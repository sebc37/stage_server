/*******************************************************************************
 * goy_lib.c  –  GOY shell model – Adams-Bashforth 2nd order integrator
 *
 * Exported API (callable from Python via ctypes):
 *
 *   goy_init(params)
 *       Initialise the model from a GoyParams struct.
 *
 *   goy_integrate(Xpp, Ypp, Xp, Yp, n_steps,
 *                 out_Xp, out_Yp, out_X, out_Y)
 *       Integrate n_steps from state (Xpp,Ypp) at t-dt and (Xp,Yp) at t.
 *       Returns the two last states:
 *           (out_Xp, out_Yp)  = state at  t + (n_steps-1)*dt
 *           (out_X,  out_Y)   = state at  t +  n_steps   *dt
 *
 * All arrays have length N (set at compile time, default 22).
 * Compile:
 *   gcc -O2 -shared -fPIC -o goy_lib.so goy_lib.c -lm
 ******************************************************************************/

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── compile-time default ── */
#ifndef N_MAX
#define N_MAX 64          /* upper bound; actual N is set at runtime via params */
#endif

/* ── public parameter struct ── */
typedef struct {
    int    N;             /* number of shells                                  */
    double k0;            /* wave-number of the first shell                    */
    double lmb;           /* ratio between consecutive shell wave-numbers      */
    double eps;           /* parameter of the NL coupling                      */
    double nu;            /* viscosity                                         */
    double dt;            /* time step                                         */
    double force;         /* forcing amplitude                                 */
    int    N_force;       /* index of the forced shell                         */
    int    force_rnd;     /* 1 = random forcing, 0 = deterministic             */
} GoyParams;

/* ── internal state (set by goy_init) ── */
static GoyParams p;
static double sh   [N_MAX];   /* shell wave-numbers          */
static double A    [N_MAX];   /* viscous damping factors     */
static double A2   [N_MAX];   /* NL coefficient -eps/lmb     */
static double A3   [N_MAX];   /* NL coefficient -(1-eps)/lmb²*/
static double D    [N_MAX];   /* forcing mask                */

/* ────────────────────────────────────────────────────────────────────────── */
/* Initialisation                                                              */
/* ────────────────────────────────────────────────────────────────────────── */
void goy_init(const GoyParams *params)
{
    int i;
    p = *params;

    for (i = 0; i < p.N; i++) {
        sh[i] = p.k0 * pow(p.lmb, (double)i);
        A [i] = exp(-p.nu * sh[i]*sh[i] * p.dt);
        A2[i] = -p.eps / p.lmb;
        A3[i] = -(1.0 - p.eps) / (p.lmb * p.lmb);
        D [i] = 0.0;
    }
    if (p.N_force >= 0 && p.N_force < p.N)
        D[p.N_force] = 1.0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/* GOY non-linear term  NY  (imaginary part coupling)                         */
/* ────────────────────────────────────────────────────────────────────────── */
static void compute_NY_GOY(const double *ax, const double *ay, double *res)
{
    int k;
    int n = p.N;

    res[0] = ax[2]*ax[1] - ay[1]*ay[2];
    res[1] = ax[3]*ax[2] - ay[2]*ay[3]
           + A2[1] * (ax[2]*ax[0] - ay[2]*ay[0]);

    for (k = 2; k < n-2; k++)
        res[k] = ax[k+2]*ax[k+1] - ay[k+1]*ay[k+2]
               + A2[k] * (ax[k+1]*ax[k-1] - ay[k+1]*ay[k-1])
               + A3[k] * (ax[k-1]*ax[k-2] - ay[k-1]*ay[k-2]);

    res[n-2] = A2[n-2] * (ax[n-1]*ax[n-3] + ay[n-1]*ay[n-3])
             + A3[n-2] * (ax[n-3]*ax[n-4] + ay[n-3]*ay[n-4]);

    res[n-1] = A3[n-1] * (ax[n-2]*ax[n-3] - ay[n-2]*ay[n-3]);

    for (k = 0; k < n; k++) res[k] *= sh[k];
}

/* ────────────────────────────────────────────────────────────────────────── */
/* GOY non-linear term  NX  (real part coupling)                              */
/* ────────────────────────────────────────────────────────────────────────── */
static void compute_NX_GOY(const double *ax, const double *ay, double *res)
{
    int k;
    int n = p.N;

    res[0] = ax[2]*ay[1] + ay[2]*ax[1];
    res[1] = ax[3]*ay[2] + ay[3]*ax[2]
           + A2[1] * (ax[2]*ay[0] + ay[2]*ax[0]);

    for (k = 2; k < n-2; k++)
        res[k] = (ax[k+2]*ay[k+1] + ay[k+2]*ax[k+1])
               + A2[k] * (ax[k+1]*ay[k-1] + ay[k+1]*ax[k-1])
               + A3[k] * (ax[k-1]*ay[k-2] + ay[k-1]*ax[k-2]);

    res[n-2] = A2[n-2] * (ax[n-1]*ay[n-3] + ay[n-1]*ax[n-3])
             + A3[n-2] * (ax[n-3]*ay[n-4] + ay[n-3]*ax[n-4]);

    res[n-1] = A3[n-1] * (ax[n-2]*ay[n-3] + ay[n-2]*ax[n-3]);

    for (k = 0; k < n; k++) res[k] *= sh[k];
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Adams-Bashforth 2nd-order step                                              */
/*   (Xpp,Ypp) = state at t-dt    NXpp/NYpp = NL term at t-dt                */
/*   (Xp, Yp)  = state at t       NXp /NYp  = NL term at t                   */
/*   -> (X, Y) = state at t+dt                                                */
/* ────────────────────────────────────────────────────────────────────────── */
static void ab2_step(
        const double *Xp,  const double *Yp,
        const double *NXp, const double *NYp,
        const double *Xpp_in, const double *Ypp_in,   /* only for A[i]*A[i] */
        const double *NXpp, const double *NYpp,
        double f1, double f2,
        double *X_out, double *Y_out)
{
    int i;
    double dt = p.dt;

    for (i = 0; i < p.N; i++) {
        X_out[i] = A[i]*Xp[i]
                 + 1.5*dt*A[i]*NXp[i]
                 - 0.5*dt*A[i]*A[i]*NXpp[i]
                 + dt*f1*D[i];

        Y_out[i] = A[i]*Yp[i]
                 + 1.5*dt*A[i]*NYp[i]
                 - 0.5*dt*A[i]*A[i]*NYpp[i]
                 + dt*f2*D[i];
    }
    (void)Xpp_in; (void)Ypp_in; /* unused – kept for API clarity */
}

/* ────────────────────────────────────────────────────────────────────────── */
/* Main integration routine                                                    */
/* ────────────────────────────────────────────────────────────────────────── */
/**
 * goy_integrate
 *
 * @param Xpp_in, Ypp_in   State at  t - dt          (length N, read-only)
 * @param Xp_in,  Yp_in    State at  t               (length N, read-only)
 * @param n_steps           Number of time steps to perform
 * @param out_Xp, out_Yp   Output: state at t + (n_steps-1)*dt
 * @param out_X,  out_Y    Output: state at t +  n_steps   *dt
 *
 * Returns 0 on success, -1 if a NaN is detected.
 */
int goy_integrate(
        const double *Xpp_in, const double *Ypp_in,
        const double *Xp_in,  const double *Yp_in,
        int n_steps,
        double *out_Xp, double *out_Yp,
        double *out_X,  double *out_Y)
{
    int i, step;
    int n = p.N;

    /* working buffers (allocate on the stack – N ≤ 64) */
    double Xpp[N_MAX], Ypp[N_MAX];
    double Xp [N_MAX], Yp [N_MAX];
    double X  [N_MAX], Y  [N_MAX];
    double NXpp[N_MAX], NYpp[N_MAX];
    double NXp [N_MAX], NYp [N_MAX];
    double NX  [N_MAX], NY  [N_MAX];

    /* seed from input */
    memcpy(Xpp, Xpp_in, n * sizeof(double));
    memcpy(Ypp, Ypp_in, n * sizeof(double));
    memcpy(Xp,  Xp_in,  n * sizeof(double));
    memcpy(Yp,  Yp_in,  n * sizeof(double));

    /* compute initial NL terms */
    compute_NX_GOY(Xpp, Ypp, NXpp);
    compute_NY_GOY(Xpp, Ypp, NYpp);
    compute_NX_GOY(Xp,  Yp,  NXp);
    compute_NY_GOY(Xp,  Yp,  NYp);

    for (step = 0; step < n_steps; step++) {
        double f1, f2;

        if (p.force_rnd) {
            f1 = p.force * drand48();
            f2 = p.force * drand48();
        } else {
            f1 = p.force;
            f2 = p.force;
        }

        /* Adams-Bashforth step: (Xpp,Ypp) + (Xp,Yp) → X,Y */
        ab2_step(Xp, Yp, NXp, NYp, Xpp, Ypp, NXpp, NYpp, f1, f2, X, Y);

        /* NaN guard */
        if (X[0] != X[0]) return -1;

        /* compute NL at new state */
        compute_NX_GOY(X, Y, NX);
        compute_NY_GOY(X, Y, NY);

        /* rotate buffers: pp ← p, p ← current */
        memcpy(Xpp,  Xp,  n * sizeof(double));
        memcpy(Ypp,  Yp,  n * sizeof(double));
        memcpy(NXpp, NXp, n * sizeof(double));
        memcpy(NYpp, NYp, n * sizeof(double));

        memcpy(Xp,  X,  n * sizeof(double));
        memcpy(Yp,  Y,  n * sizeof(double));
        memcpy(NXp, NX, n * sizeof(double));
        memcpy(NYp, NY, n * sizeof(double));
    }

    /* output the last two states */
    memcpy(out_Xp, Xpp, n * sizeof(double));   /* t + (n_steps-1)*dt */
    memcpy(out_Yp, Ypp, n * sizeof(double));
    memcpy(out_X,  Xp,  n * sizeof(double));   /* t +  n_steps*dt    */
    memcpy(out_Y,  Yp,  n * sizeof(double));

    return 0;
}
