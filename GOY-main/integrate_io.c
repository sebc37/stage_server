#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "integrate_io.h"
#include "integrate.h"
#include "stats_io.h"

#define BCK_DIR "tmp"

char filename[256];

void read_fields()
{   int i;
    FILE *fp;

    sprintf(filename, "%s/fields", BCK_DIR);
    fp = fopen(filename,"r");
    for (i=0; i<N; i++)
    {   fscanf(fp, "%lg %lg", &(Xp[i]), &(Yp[i]));
        fscanf(fp, "%lg %lg", &(Xpp[i]), &(Ypp[i]));
    }  
    fclose(fp);
  
    compute_NX(Xpp, Ypp, NXpp);
    compute_NY(Xpp, Ypp, NYpp);
  
    compute_NX(Xp, Yp, NXp);
    compute_NY(Xp, Yp, NYp);

    sprintf(filename, "%s/N_steps", BCK_DIR);
    fp = fopen(filename, "r");
    fscanf(fp, "%d", &N_steps);
    fclose(fp);
    
    if (DO_STATS) read_stats();
    if (DO_FIR) { reset_FIR(Xf);  reset_FIR(Yf); reset_FIR(flux_f); }
}


void save_fields()
{   int i;
    FILE *fp;

    sprintf(filename, "%s/fields", BCK_DIR);
    fp = fopen(filename,"w");
    for(i=0; i<N; i++)
    { 
        fprintf(fp, "%g %g ", Xp[i], Yp[i]);
        fprintf(fp, "%g %g ", Xpp[i], Ypp[i]);
    }  
    fclose(fp);

    sprintf(filename, "%s/N_steps", BCK_DIR);
    fp = fopen(filename, "w");
    fprintf(fp, "%d ", N_steps);
    fclose(fp);
    
    if (DO_STATS) save_stats();
}


// to reset time traces files
void reset_data()
{   FILE *file;

    sprintf(filename, "%s/%s", DATA_DIR, DATA_FILE);
    file = fopen(filename, "w");
    fclose(file);
    
    sprintf(filename, "%s/%s", DATA_DIR, RECONSTRUCTED_FILE);
    file = fopen(filename, "w");
    fclose(file);
    
    sprintf(filename, "%s/%s", DATA_DIR, FLUX_DATA_FILE);
    file = fopen(filename, "w");
    fclose(file);
}


void save_data(double *my_X, double *my_Y)
{   int i;
    FILE *file;

    sprintf(filename, "%s/%s", DATA_DIR, DATA_FILE);
    file = fopen(filename, "a");
    for (i=0; i<N; i++) fprintf(file, "%g %g ", my_X[i], my_Y[i]);
    fprintf(file, "\n");
    fclose(file);
}


void save_reconstructed(double *my_X, double *my_Y, double t)
{   int i;
    double v=0;
    FILE *file;

    for (i=0; i<N; i++) v += my_X[i]*cos(sh[i]*t) - my_Y[i]*sin(sh[i]*t);
    sprintf(filename, "%s/%s", DATA_DIR, RECONSTRUCTED_FILE);
    file = fopen(filename, "a");
    fprintf(file, "%g \n", v);
    fclose(file);
}