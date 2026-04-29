#include <math.h>
#include <stdlib.h>

#include "integrate.h"
#include "stats_io.h"   // 2023-04-05: just for the time trace of the fluxes

double sh[N];

/* champ complexe (X,Y) */
double X[N], Xp[N], Xpp[N];
double Y[N], Yp[N], Ypp[N];

/* same, but LP-filtered with FIR */
double Xf[N];
double Yf[N];

/* terme non-lineaire */
double NXp[N], NXpp[N];
double NYp[N], NYpp[N];

double F[N]; /* initial forcing */
double A[N]; /* coefficients du schema numerique */
double A1[N], A2[N], A3[N]; /* coefficients du terme non lineaire */
double D[N]; /* echelle de forcage */

int N_steps;
int N_fs=(int)(1./dt/fs);


void init_NL()
{   int i;

    for (i=0; i<N; i++) A1[i] = 1.0; 
    for (i=0; i<N; i++) A2[i] = -eps/lmb;
    for (i=0; i<N; i++) A3[i] = -(1.0-eps)/(lmb*lmb);
}

void init_forcing()
{   int i;

    for (i=0; i<N; i++) D[i] = 0.0;
    D[N_force] = 1; 
}

void init_shell()
{   int i;

    for(i=0; i<N; i++) sh[i] = k0 * pow(lmb, (double)i); // wave number of shell i
    for(i=0; i<N; i++) A[i] = (double) exp(-nu*sh[i]*sh[i]*dt); //
}
 
void init_fields()
{   int i;

    for (i=0; i<N; i++)
	{   F[i] = pow(sh[i], -1./3.);;
	    Xpp[i] = F[i];
	    Ypp[i] = 1.e-4;
    }
    compute_NX(Xpp, Ypp, NXpp);
    compute_NY(Xpp, Ypp, NYpp);
      
    for(i=0; i<N; i++)
    {   Xp[i] = A[i]*(Xpp[i] + dt*NXpp[i]);
	    Yp[i] = A[i]*(Ypp[i] + dt*NYpp[i]);
    } 
    compute_NX(Xp, Yp, NXp);
    compute_NY(Xp, Yp, NYp);

    if (DO_FIR) { reset_FIR(Xf);  reset_FIR(Yf); reset_FIR(flux_f); }
    N_steps=0;
}

void reset_FIR(double *x)
{   int i;

    for(i=0; i<N; i++)  x[i]=0.;
}

void normalize_FIR(double *x)
{   int i;

    for(i=0; i<N; i++)  x[i]/=N_fs;
}


void compute_NY_GOY(double *ax, double *ay, double *res)
{   int k;

    res[0] = ax[2]*ax[1] - ay[1]*ay[2];
    res[1] = ax[3]*ax[2] - ay[2]*ay[3] + A2[1] * (ax[2]*ax[0] - ay[2]*ay[0]);
  
    for(k=2; k< N-2; k++)
        res[k] = ax[k+2]*ax[k+1] - ay[k+1]*ay[k+2] 
                + A2[k] * (ax[k+1]*ax[k-1] - ay[k+1]*ay[k-1]) 
                + A3[k] * (ax[k-1]*ax[k-2] - ay[k-1]*ay[k-2]);
  
    res[N-2] = A2[N-2] * (ax[N-1]*ax[N-3] + ay[N-1]*ay[N-3]) 
             + A3[N-2] * (ax[N-3]*ax[N-4] + ay[N-3]*ay[N-4]);
  
    res[N-1] = A3[N-1] * (ax[N-2]*ax[N-3] - ay[N-2]*ay[N-3]);  
    
    for(k=0; k< N; k++) res[k] = res[k] * sh[k];
}


void compute_NX_GOY(double *ax, double *ay, double *res)
{   int k;

    res[0] = ax[2]*ay[1] + ay[2]*ax[1];
    res[1] = ax[3]*ay[2] + ay[3]*ax[2] + A2[1] * (ax[2]*ay[0] + ay[2]*ax[0]);

    for(k=2; k < N-2; k++)
        res[k] = (ax[k+2]*ay[k+1] + ay[k+2]*ax[k+1])
                + A2[k] * (ax[k+1]*ay[k-1] + ay[k+1]*ax[k-1])
                + A3[k] * (ax[k-1]*ay[k-2] + ay[k-1]*ax[k-2]);

    res[N-2] = A2[N-2] * (ax[N-1]*ay[N-3] + ay[N-1]*ax[N-3])
             + A3[N-2] * (ax[N-3]*ay[N-4] + ay[N-3]*ax[N-4]);
  
    res[N-1] = A3[N-1] * (ax[N-2]*ay[N-3] + ay[N-2]*ax[N-3]);
  
    for(k=0; k < N; k++) res[k] = res[k] * sh[k];
}


void compute_NY_Sabra(double *ax, double *ay, double *res)
{   int k;

    res[0] = ax[2]*ax[1] - ay[1]*ay[2];
    res[1] = ax[3]*ax[2] - ay[2]*ay[3] + A2[1] * (ax[2]*ax[0] - ay[2]*ay[0]);
  
    for(k=2; k< N-2; k++)
        res[k] = ax[k+2]*ax[k+1] - ay[k+1]*ay[k+2] 
                + A2[k] * (ax[k+1]*ax[k-1] - ay[k+1]*ay[k-1]) 
                + A3[k] * (ax[k-1]*ax[k-2] - ay[k-1]*ay[k-2]);
  
    res[N-2] = A2[N-2] * (ax[N-1]*ax[N-3] + ay[N-1]*ay[N-3]) 
             + A3[N-2] * (ax[N-3]*ax[N-4] + ay[N-3]*ay[N-4]);
  
    res[N-1] = A3[N-1] * (ax[N-2]*ax[N-3] - ay[N-2]*ay[N-3]);  
    
    for(k=0; k< N; k++) res[k] = res[k] * sh[k];
}


void compute_NX_Sabra(double *ax, double *ay, double *res)
{   int k;

    res[0] = ax[2]*ay[1] + ay[2]*ax[1];
    res[1] = ax[3]*ay[2] + ay[3]*ax[2] + A2[1] * (ax[2]*ay[0] + ay[2]*ax[0]);

    for(k=2; k < N-2; k++)
        res[k] = (ax[k+2]*ay[k+1] + ay[k+2]*ax[k+1])
                + A2[k] * (ax[k+1]*ay[k-1] + ay[k+1]*ax[k-1])
                + A3[k] * (ax[k-1]*ay[k-2] + ay[k-1]*ax[k-2]);

    res[N-2] = A2[N-2] * (ax[N-1]*ay[N-3] + ay[N-1]*ax[N-3])
             + A3[N-2] * (ax[N-3]*ay[N-4] + ay[N-3]*ax[N-4]);
  
    res[N-1] = A3[N-1] * (ax[N-2]*ay[N-3] + ay[N-2]*ax[N-3]);
  
    for(k=0; k < N; k++) res[k] = res[k] * sh[k];
}


void integrate()
{   int i;
    double force1, force2;

    if (force_rnd)
    {   force1 = force*drand48();
        force2 = force*drand48();
    }
    else 
    {   force1 = force;
        force2 = force;
    }

    for(i=0; i<N; i++)
    {   X[i] = A[i]*Xp[i] + 1.5*dt*A[i]*NXp[i] - 0.5*dt*A[i]*A[i]*NXpp[i] + dt*force1*D[i];
        Y[i] = A[i]*Yp[i] + 1.5*dt*A[i]*NYp[i] - 0.5*dt*A[i]*A[i]*NYpp[i] + dt*force2*D[i];
    }
    
    if (DO_FIR)
    {   for(i=0; i<N; i++)
        {   Xf[i] += X[i];
            Yf[i] += Y[i];
        }
    }
}

