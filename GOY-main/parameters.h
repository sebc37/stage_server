/****************************************************************************/
/* this file contains all adaptable parameters                              */
/* Nicolas B. Garnier 2022-07-05                                            */
/****************************************************************************/
#define GOY_MODEL   // main model
// #define SABRA_MODEL // other possibility (WIP)

/****************************************************************************/
// model parameters:
#define force 0.005 // strength of the random forcing
#define N_force 4   // mode which is forced
#define force_rnd 1 // random forcing (1) or deterministic (0)
#define k0 0.125    // largest scale
#define lmb 2.0     // ratio between consecutive scales
#define eps 0.5     // for the NL coefficients
#define nu 1.e-7    // viscosity
#define N 22        // nb of modes

/****************************************************************************/
// scheme parameters:
#define dt 1.e-5    // for integration scheme 
#define fs 100.// for data saving
#define time 1000  // for data saving (you'll get Npts = time*fs per simulation)

/****************************************************************************/
// behavioral paarameters:
#define DO_CONTINUE 0                   // to continue the previous simulation
#define DO_SAVE 1                       // to save time trace of shells
#define DO_FLUX 0                       // to save time trace of fluxes between shells
#define DO_MOMENTS 0                    // to compute and save moments
#define DO_STATS 0                      // to compute additional statistics (useless)
#define DO_RECONSTRUCT 1                // to output a "reconstructed" velocity signal
#define DO_FIR 0						// to low-pass filter time traces (anti-aliasing)
										//	-> this impacts the ouputs of DO_SAVE and DO_RECONSTRUCT

#define MOMENTS_ORDER_MAX 6             // max order of the moments to compute
#define MOMENTS_FILE "moments.dat"      // file name for the moments
#define DATA_FILE "data.dat"            // where to save the main output (shells) (from DO_SAVE )
#define RECONSTRUCTED_FILE "v.dat"      // reconstructed velocity field (from DO_RECONSTRUCT)
#define STATS_DIR "stats"               // where to save additional stats
#define DATA_DIR "."                    // where to save data

