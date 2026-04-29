#include "parameters.h"

#define PI M_PI
extern double sh[N];

/* champ complexe (X,Y) */
extern double X[N], Xp[N], Xpp[N], Xf[N];
extern double Y[N], Yp[N], Ypp[N], Yf[N];

/* terme non-lineaire */
extern double NXp[N], NXpp[N];
extern double NYp[N], NYpp[N];

extern double F[N]; /* initial forcing */
extern double A[N]; /* coefficients du schema numerique */
extern double A1[N], A2[N], A3[N]; /* coefficients du terme non lineaire */
extern double D[N]; /* echelle de forcage */

extern int N_steps;	// nb of points that have been output at the required sampling frequency fs
extern int N_fs;	// nb of iteration steps in-between 2 successive ouput points (undersampling)

// if ever we want to use the Sabra model (indeed, useless)
#ifdef SABRA_MODEL
#define compute_NX compute_NX_Sabra
#define compute_NY compute_NY_Sabra
#else
#define compute_NX compute_NX_GOY
#define compute_NY compute_NY_GOY
#endif

void init_NL();
void init_forcing();
void init_shell();
void init_fields();
void reset_FIR(double *x);
void normalize_FIR(double *x);

void compute_NX_GOY();
void compute_NY_GOY();
void compute_NX_Sabra();
void compute_NY_Sabra();
void integrate(); 

