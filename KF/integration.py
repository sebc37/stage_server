import numpy as np
import os
from pathlib import Path


# ============================================================================
#                            PARAMETERS
# ============================================================================

class GOYParams:
    """Configuration parameters for the GOY model."""
    def __init__(self,force,N_force,force_rnd,k0,lmb,eps,nu,N,dt,fs,time):
        
    # Model parameters
        self.force = force #0.005              # strength of the random forcing
        self.N_force = N_force# 4                # mode which is forced
        self.force_rnd = force_rnd #True           # random forcing (True) or deterministic (False)
        self.k0 = k0 #0.125                 # largest scale
        self.lmb = lmb #2.0                  # ratio between consecutive scales
        self.eps = eps #0.5                  # for the NL coefficients
        self.nu = nu #1.0e-7                # viscosity
        self.N = N #22                     # nb of modes
        self.DO_FIR = True
    # Scheme parameters
        self.dt = dt #1.0e-5                # for integration scheme
        self.fs = fs #100.                 # for data saving (frequency)
        self.time = time #1.0e3               # total simulation time
    xp=1
    
class GOYShellModel:
    """The GOY shell model integrator."""
    
    def __init__(self, params=None,Xp=None,Yp=None,Xpp=None,Ypp=None):
        """Initialize the GOY shell model.
        
        Parameters
        ----------
        params : GOYParams, optional
            Model parameters. If None, uses default parameters.
        """
        self.p = params if params is not None else GOYParams()
        
        # Derived parameters
        self.N_fs = int(1.0 / (self.p.dt * self.p.fs))
        self.N_steps = 0
        
        # Initialize arrays
        self.sh = np.zeros(self.p.N)        # wave numbers
        self.X = np.zeros(self.p.N)         # real part of complex amplitude
        self.Y = np.zeros(self.p.N)         # imaginary part
        if type(Xp)==type(None):
            self.Xp = np.zeros(self.p.N)        # X at previous timestep
        else:
            self.Xp = Xp
        if type(Xpp)==type(None):
            self.Xpp = np.zeros(self.p.N)       # X at two timesteps ago
        else:
            self.Xpp = Xpp
        if type(Yp)==type(None):
            self.Yp = np.zeros(self.p.N)        # Y at previous timestep
        else:
            self.Yp = Yp
        if type(Ypp)==type(None):
            self.Ypp = np.zeros(self.p.N)       # Y at two timesteps ago
        else:
            self.Ypp = Ypp
        
        # Low-pass filtered versions (FIR)
        self.Xf = np.zeros(self.p.N)
        self.Yf = np.zeros(self.p.N)
        
        # Non-linear terms
        self.NXp = np.zeros(self.p.N)
        self.NXpp = np.zeros(self.p.N)
        self.NYp = np.zeros(self.p.N)
        self.NYpp = np.zeros(self.p.N)
        
        # Coefficients and forcing
        self.A = np.zeros(self.p.N)         # numerical scheme coefficients
        self.A1 = np.ones(self.p.N)         # NL coefficient (always 1)
        self.A2 = np.full(self.p.N, -self.p.eps / self.p.lmb)      # NL coefficient
        self.A3 = np.full(self.p.N, -(1.0 - self.p.eps) / (self.p.lmb**2))  # NL coefficient
        self.F = np.zeros(self.p.N)         # initial forcing profile
        self.D = np.zeros(self.p.N)         # forcing amplitude distribution
        
        # File handles for output
        self.data_file = None
        self.reconstructed_file = None
        self.flux_file = None
        
        # Initialize all parameters
     
        self._init_forcing()
        self._init_shell()

    
    def _init_forcing(self):
        """Initialize forcing distribution."""
        self.D[:] = 0.0
        self.D[self.p.N_force] = 1.0
    
    def _init_shell(self):
        """Initialize shell wave numbers and numerical scheme coefficients."""
        for i in range(self.p.N):
            self.sh[i] = self.p.k0 * (self.p.lmb ** i)
            self.A[i] = np.exp(-self.p.nu * self.sh[i]**2 * self.p.dt)
    
    def init_fields(self):
        """Initialize the fields with default initial conditions."""
        for i in range(self.p.N):
            self.F[i] = self.sh[i] ** (-1.0/3.0)
            self.Xpp[i] = self.F[i]
            self.Ypp[i] = 1.0e-4
        
        self.compute_NX(self.Xpp, self.Ypp, self.NXpp)
        self.compute_NY(self.Xpp, self.Ypp, self.NYpp)
        
        for i in range(self.p.N):
            self.Xp[i] = self.A[i] * (self.Xpp[i] + self.p.dt * self.NXpp[i])
            self.Yp[i] = self.A[i] * (self.Ypp[i] + self.p.dt * self.NYpp[i])
        
        self.compute_NX(self.Xp, self.Yp, self.NXp)
        self.compute_NY(self.Xp, self.Yp, self.NYp)
        
        if self.p.DO_FIR:
            self.reset_FIR(self.Xf)
            self.reset_FIR(self.Yf)
        
        self.N_steps = 0
    
    def reset_FIR(self, x):
        """Reset FIR filter array."""
        x[:] = 0.0
    
    def normalize_FIR(self, x):
        """Normalize FIR filter array."""
        x[:] /= self.N_fs
    
    def compute_NY_GOY(self, ax, ay):
        """Compute the Y component of the non-linear term for GOY model.
        
        Parameters
        ----------
        ax : ndarray
            X component (real part) at current timestep
        ay : ndarray
            Y component (imaginary part) at current timestep
        
        Returns
        -------
        res : ndarray
            Time derivative of Y
        """
        res = np.zeros(self.p.N)
        
        res[0] = ax[2] * ax[1] - ay[1] * ay[2]
        res[1] = (ax[3] * ax[2] - ay[2] * ay[3] + 
                  self.A2[1] * (ax[2] * ax[0] - ay[2] * ay[0]))
        
        for k in range(2, self.p.N - 2):
            res[k] = (ax[k+2] * ax[k+1] - ay[k+1] * ay[k+2] +
                      self.A2[k] * (ax[k+1] * ax[k-1] - ay[k+1] * ay[k-1]) +
                      self.A3[k] * (ax[k-1] * ax[k-2] - ay[k-1] * ay[k-2]))
        
        res[self.p.N - 2] = (self.A2[self.p.N - 2] * (ax[self.p.N - 1] * ax[self.p.N - 3] + 
                                                       ay[self.p.N - 1] * ay[self.p.N - 3]) +
                             self.A3[self.p.N - 2] * (ax[self.p.N - 3] * ax[self.p.N - 4] + 
                                                       ay[self.p.N - 3] * ay[self.p.N - 4]))
        
        res[self.p.N - 1] = self.A3[self.p.N - 1] * (ax[self.p.N - 2] * ax[self.p.N - 3] - 
                                                       ay[self.p.N - 2] * ay[self.p.N - 3])
        
        res *= self.sh
        return res
    
    def compute_NX_GOY(self, ax, ay):
        """Compute the X component of the non-linear term for GOY model.
        
        Parameters
        ----------
        ax : ndarray
            X component (real part) at current timestep
        ay : ndarray
            Y component (imaginary part) at current timestep
        
        Returns
        -------
        res : ndarray
            Time derivative of X
        """
        res = np.zeros(self.p.N)
        
        res[0] = ax[2] * ay[1] + ay[2] * ax[1]
        res[1] = (ax[3] * ay[2] + ay[3] * ax[2] + 
                  self.A2[1] * (ax[2] * ay[0] + ay[2] * ax[0]))
        
        for k in range(2, self.p.N - 2):
            res[k] = ((ax[k+2] * ay[k+1] + ay[k+2] * ax[k+1]) +
                      self.A2[k] * (ax[k+1] * ay[k-1] + ay[k+1] * ax[k-1]) +
                      self.A3[k] * (ax[k-1] * ay[k-2] + ay[k-1] * ax[k-2]))
        
        res[self.p.N - 2] = (self.A2[self.p.N - 2] * (ax[self.p.N - 1] * ay[self.p.N - 3] + 
                                                       ay[self.p.N - 1] * ax[self.p.N - 3]) +
                             self.A3[self.p.N - 2] * (ax[self.p.N - 3] * ay[self.p.N - 4] + 
                                                       ay[self.p.N - 3] * ax[self.p.N - 4]))
        
        res[self.p.N - 1] = self.A3[self.p.N - 1] * (ax[self.p.N - 2] * ay[self.p.N - 3] + 
                                                       ay[self.p.N - 2] * ax[self.p.N - 3])
        
        res *= self.sh
        return res
    
    def compute_NX(self, ax, ay, res):
        """Compute non-linear X term in-place."""
        res[:] = self.compute_NX_GOY(ax, ay)
    
    def compute_NY(self, ax, ay, res):
        """Compute non-linear Y term in-place."""
        res[:] = self.compute_NY_GOY(ax, ay)
    
    def integrate(self):
        """Perform one integration step using the numerical scheme."""
        if self.p.force_rnd:
            force1 = self.p.force * np.random.random()
            force2 = self.p.force * np.random.random()
        else:
            force1 = self.p.force
            force2 = self.p.force
        
        for i in range(self.p.N):
            self.X[i] = (self.A[i] * self.Xp[i] + 
                         1.5 * self.p.dt * self.A[i] * self.NXp[i] -
                         0.5 * self.p.dt * self.A[i]**2 * self.NXpp[i] +
                         self.p.dt * force1 * self.D[i])
            
            self.Y[i] = (self.A[i] * self.Yp[i] +
                         1.5 * self.p.dt * self.A[i] * self.NYp[i] -
                         0.5 * self.p.dt * self.A[i]**2 * self.NYpp[i] +
                         self.p.dt * force2 * self.D[i])
        
        if self.p.DO_FIR:
            self.Xf += self.X
            self.Yf += self.Y
    

    
    def run(self):
        """Run the full simulation."""
        count = int(self.p.time / self.p.dt)
        if self.p.xp is None:
            self.init_fields()
        
        
        # print(f"Starting GOY model integration: {count} steps ({self.p.time} time units)")
        # print(f"Output interval: {self.N_fs} steps = {1.0/self.p.fs} time units")
        
        while count > 0:
            self.integrate()
            
            # if (count % self.N_fs) == 0:
            #     # Print progress
            #     remaining_steps = int(count / self.N_fs)
            #     if not np.isnan(self.X[0]):
            #         print(f"Remaining steps: {remaining_steps}")
            #     else:
            #         print(f"Remaining steps: {remaining_steps} : ERROR (NaN detected)")
                
            #     # Save data
            #     if self.p.DO_FIR:
            #         self.normalize_FIR(self.Xf)
            #         self.normalize_FIR(self.Yf)
                    
                 
            #         self.reset_FIR(self.Xf)
            #         self.reset_FIR(self.Yf)
            
            #     self.N_steps += 1
            
            # Evolve fields
            self.Xpp[:] = self.Xp
            self.Ypp[:] = self.Yp
            self.NXpp[:] = self.NXp
            self.NYpp[:] = self.NYp
            self.Xp[:] = self.X
            self.Yp[:] = self.Y
            
            self.compute_NX(self.Xp, self.Yp, self.NXp)
            self.compute_NY(self.Xp, self.Yp, self.NYp)
            
            count -= 1
        
        #print("Integration completed!")
        return self.X,self.Y

