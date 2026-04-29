#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "integrate.h"
#include "stats_io.h"

double Mp[MOMENTS_ORDER_MAX][N];
double Mp_three[MOMENTS_ORDER_MAX][N];
double Mp_flux[MOMENTS_ORDER_MAX][N];
double flux[N];         // flux from k to k+1
double flux_f[N];       // same but LP-filtered with FIR
double flux_mean[N];
long int N_stats;

extern char filename[256];

void compute_flux()
{   double mod_m, mod_p;
    int i;
    /* flux */
    for(i=0; i<N-1; i++)
    {
	    if (i==0)       mod_m = 0;
	    else            mod_m = Y[i+1]*(X[i-1]*X[i]-Y[i-1]*Y[i]) + X[i+1]*(X[i-1]*Y[i]+Y[i-1]*X[i]);
	    
	    if (i==(N-2))   mod_p = 0;
	    else            mod_p = Y[i+2]*(X[i]*X[i+1]-Y[i]*Y[i+1]) + X[i+2]*(X[i]*Y[i+1]+Y[i]*X[i+1]);
	
      	flux[i] = -sh[i]*(mod_p + (1-eps)/lmb*mod_m);
      	
      	if (DO_FIR)     flux_f[i] += flux[i];
    }
}

void save_flux(double *x)
{   FILE *file;
    int i;
    
    sprintf(filename, "%s/%s", DATA_DIR, FLUX_DATA_FILE);
    file = fopen(filename, "a");
    for (i=0; i<N-1; i++) fprintf(file, "%g ", x[i]);
    fprintf(file, "\n");
    fclose(file);
}

void init_moments()
{   FILE *file;

    sprintf(filename, "%s/%s", STATS_DIR, MOMENTS_FILE);
    file = fopen(filename, "w");
    fclose(file);
    
    N_stats=0;
}

void save_moments(double *my_X, double *my_Y)
{   int i, k;
    double mod, val;
    FILE *file;

    /* moments classiques */
    for (i=0; i<N; i++)
    {   mod = sqrt(my_X[i]*my_X[i] + my_Y[i]*my_Y[i]);
	    val = mod;
	    for (k=0; k<MOMENTS_ORDER_MAX; k++)
	    { Mp[k][i] += val;
	      val = val*mod;
	    }
    }

    sprintf(filename, "%s/%s", STATS_DIR, MOMENTS_FILE);
    file = fopen(filename, "at");
    for(i=0; i<N; i++) 
    {   for (k=0; k<MOMENTS_ORDER_MAX; k++)  fprintf(file, "%g ", Mp[k][i]);
    }
    fprintf(file, "\n");
    fclose(file);
}

void save_stats()
{   int i, k;
    FILE *fp;
    double mod, val;
    double ilmb2;

    /* produit de trois couches successives */
    for(i = 0; i < N; i++)
	{   if ((i==0) || (i==(N-1))) mod = sqrt(X[i]*X[i] + Y[i]*Y[i]);
	    else  mod = pow((X[i-1]*X[i-1] + Y[i-1]*Y[i-1]) * (X[i]*X[i] + Y[i]*Y[i]) * (X[i+1]*X[i+1] + Y[i+1]*Y[i+1]), 1./6.);
	
	    val = mod;
	    for (k=0; k<MOMENTS_ORDER_MAX; k++)
	    {   Mp_three[k][i] += val;
	        val= val*mod;
	    }
    }
    sprintf(filename, "%s/%s", STATS_DIR, MOMENTS_THREE_FILE);
    fp = fopen(filename, "w");
    for (i=0; i<N; i++)
    {   for(k=0; k<MOMENTS_ORDER_MAX; k++)  fprintf(fp, "%g ", Mp_three[k][i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
  
    /* flux */
    ilmb2 = 1.0 / (lmb*lmb);
    for(i = 0; i < N-1; i++)
    {
	    if (i==0)
	        mod = Y[i+2]*(X[i]*X[i+1]-Y[i]*Y[i+1]) + X[i+2]*(X[i]*Y[i+1]+Y[i]*X[i+1]);
	  	else if (i==(N-2))
	        mod = ilmb2 * (Y[i+1]*(X[i-1]*X[i]-Y[i-1]*Y[i]) + X[i+1]*(X[i-1]*Y[i]+Y[i-1]*X[i]));
	   	else
	   	{   mod = Y[i+2]*(X[i]*X[i+1]-Y[i]*Y[i+1]) + X[i+2]*(X[i]*Y[i+1]+Y[i]*X[i+1]);
	        mod += ilmb2 * (Y[i+1]*(X[i-1]*X[i]-Y[i-1]*Y[i]) + X[i+1]*(X[i-1]*Y[i]+Y[i-1]*X[i]));
	    }
	
      	flux_mean[i] += sh[i]*mod;
	
      	if(mod<0) mod = -mod;
	
       	mod = pow(mod, 1./3.);
	    val = mod;
	
       	for(k=0; k<MOMENTS_ORDER_MAX; k++)
       	{   Mp_flux[k][i] += val;
	        val= val*mod;
	    }
    }
    sprintf(filename, "%s/%s", STATS_DIR, FLUX_MOMENTS_FILE);
    fp = fopen(filename, "w");
    for (i=0; i<N-1; i++)
    {   for(k=0; k<MOMENTS_ORDER_MAX; k++)  fprintf(fp, "%g ", Mp_flux[k][i]);
        fprintf(fp, "\n");
    }
    fclose(fp);

    sprintf(filename, "%s/%s", STATS_DIR, FLUX_MEAN_FILE);
    fp = fopen(filename, "w");
    for (i=0; i<N-1; i++)   fprintf(fp, "%g ", flux_mean[i]);
    fclose(fp);

    N_stats++;
    sprintf(filename, "%s/N_stats", STATS_DIR);
    fp = fopen(filename, "w");
    fprintf(fp, "%ld ", N_stats);
    fclose(fp);
}

void read_stats()
{   int i, k;
    FILE *fp;

    sprintf(filename, "%s/%s", STATS_DIR, MOMENTS_FILE);
    fp = fopen(filename, "r");
    for(i = 0; i < N; i++){
      for(k=0; k<MOMENTS_ORDER_MAX; k++)
	fscanf(fp, "%lg ",&(Mp[k][i]));
      fprintf(fp, "\n");
    }
    fclose(fp);
    
    sprintf(filename, "%s/%s", STATS_DIR, MOMENTS_THREE_FILE);
    fp = fopen(filename, "r");
    for(i = 0; i < N; i++){
      for(k=0; k<MOMENTS_ORDER_MAX; k++)
	fscanf(fp, "%lg ", &(Mp_three[k][i]));
      fprintf(fp, "\n");
    }
    fclose(fp);
    
    sprintf(filename, "%s/Moments_flux", STATS_DIR);
    fp = fopen(filename, "r");
    for(i = 0; i < N-1; i++){
      for(k=0; k<MOMENTS_ORDER_MAX; k++)
	fscanf(fp, "%lg ", &(Mp_flux[k][i]));
      fprintf(fp, "\n");
    }
    fclose(fp);
    
    sprintf(filename, "%s/mean_flux", STATS_DIR);
    fp = fopen(filename, "r");
    for(i = 0; i < N-1; i++)
      fscanf(fp, "%lg ", &(flux_mean[i]));
    fclose(fp);
    
    sprintf(filename, "%s/N_stats", STATS_DIR);
    fp = fopen(filename, "r");
    fscanf(fp, "%ld ", &N_stats);
    fclose(fp);
}
