#include "parameters.h"

// extra file names:
#define MOMENTS_THREE_FILE  "moments_three.dat"
#define FLUX_DATA_FILE      "flux.dat"
#define FLUX_MEAN_FILE      "flux_mean.dat"
#define FLUX_MOMENTS_FILE   "flux_moments.dat"


extern long int N_stats;
extern double flux[N];         // flux from k to k+1
extern double flux_f[N];       // same but LP-filtered with FIR

void init_moments();
void save_moments(double *X, double *Y);
void save_stats();
void read_stats();

void compute_flux();
void save_flux(double *x);
