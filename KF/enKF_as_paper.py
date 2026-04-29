import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tqdm
from goy import GoyModel



def filter_mode(X,mode_min:int,mode_max:int,t_min:int,ratio:float,seed):
    
    np.random.seed(seed=seed)
    X_subset = np.copy(X)
    X_subset[:,0:mode_min] = None #enlève les modes inférieurs à mode_min
    X_subset[0:t_min,:] = None #enlève la phase de stabilisation
    nb_column = np.shape(X_subset)[1]
    nb_line = np.shape(X_subset)[0]
    X_filtered = np.copy(X_subset)
    X_posx = []
    X_posy = []
    X_value = []
    X_dataset = []

    var_mode = [np.var(X[:,j]) for j in range(nb_column)]
    std_mode = [np.std(X[:,j]) for j in range(nb_column)]
    mean_mode = [np.mean(X[:,j]) for j in range(nb_column)]

    for j in range(mode_min,mode_max):
        for i in range(t_min,nb_line):
            if (np.random.random()<=ratio):
                X_filtered[i,j] =  X_subset[i,j] #+ np.random.normal(0,1) 
                X_posx.append(j)
                X_posy.append(i)
                X_value.append(X_subset[i,j])#-mean_mode[j])/std_mode[j]) # centré réduit

            else:
                X_filtered[i,j] = None
    
    X_filtered[:,2*k_max_collocation:] = None
    # Y = np.zeros((nb_line,nb_column))
    # Y[:,:] = None
    # Y[X_posy,X_posx] =  X_value #X_filtered[X_posy,X_posx] # on ajoute du bruit gaussien aux observations

    pourcentage_filtered = nb_line*ratio/nb_line*100
    X_dataset.append(X_posx)
    X_dataset.append(X_posy)
    X_dataset.append(X_value)


    return X_filtered,X_dataset,mean_mode,var_mode,std_mode,pourcentage_filtered#,Y




PATH = "/Odyssey/private/s26calme/code_stage/GOY-main/"
path_data = PATH + "data_enKF_100dt.dat"
SAVE = "/Odyssey/private/s26calme/code_stage/KF/"

data =  np.loadtxt(path_data,dtype=np.float64) # charge le jeu de données
Nmax = np.shape(data)[0] # nombres de pas de temps
debut = int(0.1*Nmax) # skip la phase de stabilisation

Data_shell = data[debut:Nmax,:] # on garde  partie réelle de chaque shell
Npts = np.shape(Data_shell)[0] # nombre de pas dans le temps

# nb of shells selected for training the PINN on collocatin point
k_min_collocation = 6 
k_max_collocation = 8 

#nb of shells for training on boundary conditions
k_bc_min = 0
k_bc_max = 4 

k0 = 0.125
lmb = 2.0
# retourne un dataset pour plot , var,std,et mean pour chaque mode et les colocation point centré réduit
Data_filtered, Data_train, mean, Var_mode, Std_mode, perc= filter_mode(Data_shell,2*k_min_collocation,2*k_max_collocation,0,0.001,123456)
#Data_shell = reduced_center(Data_shell,mean=mean,std=Std_mode)
#Data_filtered = reduced_center(Data_filtered,mean=mean,std=Std_mode) # données réelles

K = np.array([k0*lmb**i for i in range(22)],dtype=np.float32)

#########################
#       calcul Tn       #
#########################

Tn = np.zeros(22)
for n in range(22):
    Tn[n] = 1/(K[n]*np.sqrt((np.mean(np.sqrt(Data_shell[:,2*n]**2 + Data_shell[:,2*n+1]**2)**2))))

Tn = Tn**2 # énergie de chaque shell dans les données
plt.figure()
plt.plot(Tn,'.b')
plt.xlabel('Shell number')
plt.ylabel('Turnover time')
plt.savefig(SAVE + "Turnover_times.png",format='png')
print("Turnover times:",min(Tn),max(Tn))
# for i in range(np.shape(y_obs)[0]):
#     plt.figure()
#     plt.plot(y_obs[i,:],'*')
#     plt.plot(Data_shell[:,i],'gray',alpha=0.5)
#     plt.savefig(PATH + "ploty",dpi=300)

shell_array = np.array(Data_shell)
MS = np.array([(0.05**2)*np.mean(shell_array[:,k]**2 ) for k in range(0,44,2)])
#MS = np.array([(0.05**2)*np.mean(shell_array[:,k]**2 + shell_array[:,k+1]**2) for k in range(44)]) # variance de l'observation pour chaque shell


### parameters
n     = 44 # state size  on veut estimer les Un de 1 à 22 avec Re et Im donc 44 variables d'état
p     = 2*(k_max_collocation-k_min_collocation + 1) # On observe Un n=5,6,7,8,9,10 avec Re et Im donc 12 variables d'observations 
nb    = Npts # number of times
time  = np.array(range(nb)) # time vector
var_Q = 0.0 # error variance of the model (in Kalman)
var_R = 0.1 # error variance of the observations (in Kalman)
x_0   = np.zeros((n)) # initial coundition (mean)
#x_0   = Data_shell[0,:] # initial coundition (mean) 
P_0   = np.eye(n,n) # initial coundition (covariance)
for i in range(22):
    P_0[2*i,2*i] = MS[i]
    P_0[2*i+1,2*i+1] = MS[i] # initial covariance = variance of each shell in the data


### variables

m = MS[4:10]
R = np.eye(p,p)
Q      = np.eye(n,n)
for i in range(22):
    Q[2*i,2*i] = var_Q*MS[i]
    Q[2*i+1,2*i+1] = var_Q*MS[i] # model covariance = variance de chaque shell observé dans les données
for i in range(int(p/2)):
    R[2*i,2*i] = var_R*m[i]
    R[2*i+1,2*i+1] = var_R*m[i] # observation covariance = variance de chaque shell observé dans les données
#R      = np.fill_diagonal(R,list(m[:]))#var_R*np.eye(p,p)

##############  noisy observations ##################
y_obs = Data_shell.T.copy() # observations = données réelles  
y_obs_ = y_obs[2*k_min_collocation:2*k_max_collocation,:]
a = y_obs_.copy()

# for t in range(Npts):
#     y_obs_[:,t]  = y_obs_[:,t] + np.random.multivariate_normal(np.zeros(p),R)
#y_obs = H @ Data_shell.T + np.random.multivariate_normal(np.zeros(p),R,size=(Npts,)).T # observations bruitées = H @ données réelles + bruit gaussien

# for i in range(p):
#     plt.figure()
#     #plt.plot(y_obs_[i,:],'.k',alpha=0.2,label='Observations')
#     plt.fill_between(time[0:nb], a[i,0:nb] - 1.96*np.sqrt(R[i,i]), a[i,0:nb] + 1.96*np.sqrt(R[i,i]), facecolor='red')
#     plt.plot(a[i,:], 'b', label='True state')
#     plt.xlabel('Time', size=20)
#     plt.ylabel(f'$U_{2*(i+k_min_collocation//2)}$ (shell {i+k_min_collocation//2})', size=20)
#     plt.legend(fontsize=20)
#     plt.savefig(SAVE + f"fig{i}enKF")



######### paramètres du modèle pour l'intégration #############################
TIME      = 1000.
DT        = 1e-5
FS        = 999.
FORCE     = 0.005
N_FORCE   = 4
FORCE_RND = 0
N_fs      = int(1.0 / DT / FS)  
model = GoyModel(dt=DT, force=FORCE, N_force=N_FORCE, force_rnd=FORCE_RND)
N = model.N  # 22
nu =1.0e-7



def m_b(x_past,N_fs,n_steps_first,start=False,second = False,custom=False):
    # ── choix du point de départ ──────────────────────────────────────────────────
    #i      = 10000   # ligne du fichier depuis laquelle on repart
     # nombre de lignes suivantes à reproduire

     # pas entre deux lignes du fichier
     # pas spéciaux pour la ligne 0 (voir run_goy.py)

    # ── reconstruction de Xpp (état à t_i - dt) ──────────────────────────────────
    # On part de la ligne i-1 et on intègre 998 pas → on arrive à t_i - dt
    if start:
        # cas particulier : ligne 0, on repart des CI
        Xpp0, Ypp0, Xp0, Yp0 = model.init_fields()
        (Xpp, Ypp), (Xp, Yp) = model.integrate(Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first - 1)
    else:
        # ligne i-1 → intègre N_fs-1 pas → arrive à t_i - dt
        if custom:
            Xpp0,Ypp0,Xp0,Yp0 = model.init_fields(Xpp=x_past[0,0::2],Ypp=x_past[0,1::2])
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first)
            return cur_Xpp,cur_Ypp,cur_Xp,cur_Yp
        if second:
            Xpp0, Ypp0, Xp0, Yp0 = model.init_fields()
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xpp0, Ypp0, Xp0, Yp0, n_steps=n_steps_first)
        else:
            Xp_prev2 = x_past[0, 0::2];  Yp_prev2 = x_past[0, 1::2]
            Xp_prev1 = x_past[1, 0::2];  Yp_prev1 = x_past[1, 1::2]
            (cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
                Xp_prev2, Yp_prev2, Xp_prev1, Yp_prev1, n_steps=N_fs - 1)

        (Xpp, Ypp), (Xp, Yp) = model.integrate(
            cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=1)
        # maintenant Xp/Yp = ref[i] à la précision machine, Xpp/Ypp = état à t_i - dt
    return Xpp,Ypp,Xp,Yp

def m_step(Xpp, Ypp, Xp, Yp):
    """
    Intègre le modèle GOY de N_fs pas.
    Entrée  : Xpp,Ypp (t-dt)  Xp,Yp (t)
    Sortie  : Xpp_new,Ypp_new (t+N_fs*dt - dt)  Xp_new,Yp_new (t+N_fs*dt)
    """
    (Xpp_new, Ypp_new), (Xp_new, Yp_new) = model.integrate(
        Xpp, Ypp, Xp, Yp, n_steps=N_fs)
    return Xpp_new, Ypp_new, Xp_new, Yp_new

############################### TEST ##############################################################

j_ = 0 #np.random.randint(3,Npts-1001)
nb_iter = 3000
count_init   = int(TIME / DT)  
n_steps_first = count_init % N_fs  
# series = np.zeros((n,nb_iter)).T
# series[0,:] = Data_shell[j_,:]
# series[1,:] = Data_shell[j_+1,:]#x_past[0,:]



# for k in range(2,nb_iter):
#     x_past = series[k-2:k,:].copy()
    
#     _,_,cur_Xp,cur_Yp =m_step(Xpp=x_past[0,0::2],Ypp=x_past[0,1::2],Xp=x_past[1,0::2],Yp=x_past[1,1::2])#m_b(x_past,N_fs=999,n_steps_first=99,custom=True) #
#     series[k, 0::2] = cur_Xp.copy()
#     series[k, 1::2] = cur_Yp.copy()
#     # print("serie:",series[k,0:2])

# Rmse = np.sqrt(np.mean(Data_shell[j_:j_+nb_iter,:]-series)**2)

# print(Rmse)
# print(j_)

# plt.figure()

# for i in range(5):
#      #plt.plot(np.abs(series[:,2*i]-Data_shell[:,2*i]))
#      plt.plot(series[:,2*i],label=f"shell_encaps_{2*i}",alpha=0.5)
#      plt.plot(Data_shell[j_:j_+nb_iter,2*i],'--',label=f"shell_reel_{2*i}")
#      plt.legend()
#      plt.show()
# plt.savefig("test")


# plt.figure()
# for i in range(10):
#      plt.plot(np.abs(base[2*i,:]-data[2*i,0:9999]))
# plt.savefig("test_2")
#####################################################################################################



### Generate observations and covariance
def generate_observations(p, H):
    y = np.zeros((p,nb))
    R = var_R*np.eye(p,p) # observation covariance
    for t in range(1,nb):
        y[:,t] = H @ y_obs[:,t] + np.random.multivariate_normal(np.zeros(p), R) # noisy observations
    #y[:,i_nan] = y[:,i_nan]*np.nan # remove observations 
    return y, R

H = np.eye(44,44) #array([[1,0,0,0], [0,1,0,0]])
H = H[2*(k_min_collocation-1):2*k_max_collocation,:] # on observe que les modes de 5 à 10 avec Re et Im donc 12 variables d'observations

i_nan = np.random.choice(Npts, size=int(0.8*Npts), replace=False) # indices des observations à supprimer

y_obs = H @ Data_shell.T + np.random.multivariate_normal(np.zeros(p),R,size=(Npts,)).T # vrai observation 
#y_obs[:,i_nan] = y_obs[:,i_nan]*np.nan

# tiré des temps au hasard pour enlever des observations 

##### check observations  

# for i in range(p):
#     plt.figure()
#     plt.plot(y_obs[i,:],marker='.',color="gray",alpha=0.5,label='Observations')
#     plt.plot(Data_shell.T[i+8,:], 'r', label='True state')
#     plt.xlabel('Time')
#     plt.ylabel("y_obs vs true state")
#     plt.legend()
# plt.show()
    #plt.savefig(SAVE + f"obs_true_state_{i}")

### Ensemble Kalman initialization
Ne = 100                   # number of ensembles
x_f_enkf = np.zeros((n,nb))   # forecast state
P_f_enkf = np.zeros((n,n,nb)) # forecast error covariance matrix
x_a_enkf = np.zeros((n,nb))   # analysed state
P_a_enkf = np.zeros((n,n,nb)) # analysed error covariance matrix
P_a_tilde = np.zeros((n,n,nb)) # analysed error covariance matrix après inflation multiplicative
### Ensemble Kalman filter
x_a_enkf_tmp = np.zeros((n,Ne)) # shell,t-2 t-1, Ne
x_f_enkf_tmp = np.zeros((n,Ne))
y_f_enkf_tmp = np.zeros((p,Ne))
# initial step

ens_Xpp = np.zeros((Ne, int(n/2))) # tous les ensembles pour Xpp et Ypp == t-2
ens_Ypp = np.zeros((Ne, int(n/2))) 
ens_Xp  = np.zeros((Ne, int(n/2))) # tous les ensembles pour Xp et Yp == t-1
ens_Yp  = np.zeros((Ne, int(n/2)))


fens_Xpp = np.zeros((Ne, int(n/2))) # tous les ensembles pour Xpp et Ypp == t-2 apres intégration par le modèle
fens_Ypp = np.zeros((Ne, int(n/2)))
fens_Xp  = np.zeros((Ne, int(n/2))) # tous les ensembles pour Xp et Yp == t-1 apres intégration par le modèle
fens_Yp  = np.zeros((Ne, int(n/2)))


# condition initiales
j_start = 2
# on récupère les etats à t-2 et t-1 après intégration du modèles avec AB sur 999 pas pour avoir une précision machine et on ajoute du bruit gaussien pour initialiser les ensembles
(cur_Xpp, cur_Ypp), (cur_Xp, cur_Yp) = model.integrate(
    Data_shell[j_start-2, 0::2], Data_shell[j_start-2, 1::2],
    Data_shell[j_start-1, 0::2], Data_shell[j_start-1, 1::2],
    n_steps=N_fs)
# (ref_Xpp, ref_Ypp), (ref_Xp, ref_Yp) = model.integrate(
#     cur_Xpp, cur_Ypp, cur_Xp, cur_Yp, n_steps=1)


amp = np.std(Data_shell, axis=0)

nb = 3000
# initialisation de l'ensemble 
for i in range(Ne):
    #x_a_enkf_tmp[:,i] = np.random.multivariate_normal(x_0, P_0)
    
    #noise = np.random.randn(n) * amp * 0.01   # perturbation 1%
    ens_Xpp[i] = cur_Xpp #+ np.random.multivariate_normal(x_0, P_0)[0::2] #+ noise[0::2] * 0.1  # Xpp varie peu
    ens_Ypp[i] = cur_Ypp #+ np.random.multivariate_normal(x_0, P_0)[1::2] #+ noise[1::2] * 0.1
    ens_Xp[i]  = cur_Xp  + np.random.multivariate_normal(x_0, P_0)[0::2] # np.sqrt(Data_shell[1,0::2]**2 )*np.cos(np.random.random()*2*np.pi)
    ens_Yp[i]  = cur_Yp + np.random.multivariate_normal(x_0, P_0)[1::2] #np.sqrt(Data_shell[1,1::2]**2)*np.sin(np.random.random()*2*np.pi) # 
    x_a_enkf_tmp[0::2,i] = ens_Xp[i].T 
    x_a_enkf_tmp[1::2,i] = ens_Yp[i].T

x_a_enkf[:,0]   = np.mean(x_a_enkf_tmp,1) # initial state
P_a_enkf[:,:,0] = np.cov(x_a_enkf_tmp)    # initial state covariance

# for j in range(22):
#     plt.figure()
    
#     plt.scatter(np.zeros(Ne),ens_Xpp[:,j],alpha=0.5,label=f'Xpp {j} ')
#     plt.scatter(np.zeros(Ne),ens_Ypp[:,j],alpha=0.5,label=f'Ypp {j} ')
#     plt.scatter(np.ones(Ne),ens_Xp[:,j],alpha=0.5,label=f'Xp {j} + noise')
#     plt.scatter(np.ones(Ne),ens_Yp[:,j],alpha=0.5,label=f'Yp {j} + noise')
#     plt.scatter(1,cur_Xp[j],color='red',label=f'Xp {j} Référence')
#     plt.scatter(1,cur_Yp[j],color='red',label=f'Yp {j} Référence')
#     plt.scatter(0,cur_Xpp[j],color='blue',label=f'Xpp {j} Référence')
#     plt.scatter(0,cur_Ypp[j],color='blue',label=f'Ypp {j} Référence')
#     plt.scatter(0,Data_shell[j_start-1, 2*(j)],marker='*',alpha= 0.5,color='black',label=f'Xpp {j} Data')
#     plt.scatter(0,Data_shell[j_start-1, 2*(j)+1],marker='*',alpha= 0.5,color='black',label=f'Ypp {j} Data')
#     plt.scatter(1,Data_shell[j_start, 2*(j)],marker='*',alpha= 0.5,color='black',label=f'Xp {j} Data')
#     plt.scatter(1,Data_shell[j_start, 2*(j)+1],marker='*',alpha= 0.5,color='black',label=f'Yp {j} Data')    
#     plt.xlabel('time')
#     plt.ylabel('Xp and Yp')
#     plt.legend()
#     plt.savefig(SAVE + f"xy_start_enKF_{j}")
lmb_inf = 0.2
g = np.zeros((n,nb))
for k in tqdm.tqdm(range(nb)): # forward in time #nb
    # prediction step
    # il faut un initialisation custom pour chaque Ne
    for i in range(Ne):
        Xpp_n, Ypp_n, Xp_n, Yp_n = m_step(
                ens_Xpp[i], ens_Ypp[i], ens_Xp[i], ens_Yp[i])

        fens_Xpp[i] = Xpp_n
        fens_Ypp[i] = Ypp_n
        fens_Xp[i]  = Xp_n
        fens_Yp[i]  = Yp_n

        x_f_enkf_tmp[0::2,i] = Xp_n.T # forward sans bruit du modèle, on suppose que le modèle est parfait
        x_f_enkf_tmp[1::2,i] = Yp_n.T

        #_,_,forward_x,forward_y  = m_b(x_a_enkf_tmp[:,i].T,N_fs=999,n_steps_first=998,custom=True)
        #x_f_enkf_tmp[0::2,i],x_f_enkf_tmp[1::2,i] = forward_x.T,forward_y.T ### A CACHER
        # Modèle supposé parfait ==> pas de bruit du modèle
        #x_f_enkf_tmp[:,i] += np.random.multivariate_normal(np.zeros(n), Q)
                                            # c'est un moins dans le papier à verifier
        y_f_enkf_tmp[:,i] = H @ x_f_enkf_tmp[:,i] + np.random.multivariate_normal(np.zeros(p), R) ### A CACHER
    
    ens_Xpp = fens_Xpp
    ens_Ypp = fens_Ypp
    ens_Xp  = fens_Xp
    ens_Yp  = fens_Yp

    P_f_enkf_tmp = np.cov(x_f_enkf_tmp) ### A CACHER
    # Kalman gain
    
    K_g = P_f_enkf_tmp @ H.T @ np.linalg.inv(H @ P_f_enkf_tmp @ H.T + R) ### A CACHER
    # update step
    if(sum(np.isfinite(y_obs[:,k]))>0):
        for i in range(Ne):
            x_a_enkf_tmp[:,i] = x_f_enkf_tmp[:,i] + K_g @ (y_obs[:,k] - y_f_enkf_tmp[:,i]) ### A CACHER
        P_a_enkf_tmp = np.cov(x_a_enkf_tmp) ### A CACHER
        
        # inflation multiplicative
        P_a_tilde[:,:,k] = (np.eye(n) - K_g @ H) @ P_f_enkf_tmp ### A CACHER
        mu_n = np.mean(x_a_enkf_tmp, axis=1) ### A CACHER

        for j in range(n):
                g[j,k] = max(1,1+lmb_inf*(P_f_enkf_tmp[j,j]-P_a_tilde[j,j,k])/P_f_enkf_tmp[j,j]) # terme d'inflation multiplicative pour chaque variable d'état ### A CACHER
        
        x_f_enkf[:,k]   = np.mean(x_f_enkf_tmp,1)
        P_f_enkf[:,:,k] = P_f_enkf_tmp
        x_a_enkf[:,k]   = g[:,k]*np.mean(x_a_enkf_tmp,1) + (1-g[:,k])*mu_n # g*  + (1-g)*mu_n
        P_a_enkf[:,:,k] = P_a_tilde[:,:,k]#P_a_enkf_tmp
        # U_tilde = g*U_tilde + (1-g)*mu_n[:,None] 

    else:
            #x_a_enkf_tmp[:,:,0] = x_a_enkf_tmp[:,:,1]
            x_a_enkf_tmp = x_f_enkf_tmp
            P_a_enkf_tmp = P_f_enkf_tmp 
    # store results
            x_f_enkf[:,k]   = np.mean(x_f_enkf_tmp,1)
            P_f_enkf[:,:,k] = P_f_enkf_tmp
            x_a_enkf[:,k]   = np.mean(x_a_enkf_tmp,1)
            P_a_enkf[:,:,k] = P_a_enkf_tmp
    
    ens_Xpp = fens_Xpp
    ens_Ypp = fens_Ypp
    ens_Xp  = x_a_enkf_tmp[0::2,:].T
    ens_Yp  = x_a_enkf_tmp[1::2,:].T


### plot trajectories (true, observed, KF, EnKF)
for i in range(N):
    if (i>=k_min_collocation-1) and (i<k_max_collocation):
        plt.figure()
        plt.fill_between(time[0:nb], x_a_enkf[2*i,0:nb] - 1.96*np.sqrt(P_a_tilde[2*i,2*i,0:nb]), x_a_enkf[2*i,0:nb] + 1.96*np.sqrt(P_a_tilde[2*i,2*i,0:nb]), facecolor='red', alpha=0.4)
        
        plt.plot(y_obs[2*i-2*(k_min_collocation-1),0:nb], '.k',alpha=0.3, label=f'Observations ($y {i+1}$)')
        plt.plot(x_a_enkf[2*i,0:nb], 'r', label=f'EnKF ($U^a {i+1}$)')
        plt.plot(Data_shell.T[2*i,j_start:nb+j_start], 'b', label=f'True state ($U_{i+1}$)')
        plt.xlabel('$time$')
        plt.ylabel(f'$\Re(U_{i+1})$')
        plt.legend()
        
    else:
        plt.figure()
        plt.fill_between(time[0:nb], x_a_enkf[2*i,0:nb] - 1.96*np.sqrt(P_a_tilde[2*i,2*i,0:nb]), x_a_enkf[2*i,0:nb] + 1.96*np.sqrt(P_a_tilde[2*i,2*i,0:nb]), facecolor='red', alpha=0.4)
        plt.plot(x_a_enkf[2*i,0:nb], 'r', label=f'EnKF ($U^a_{i+1}$)')
        plt.plot(Data_shell.T[2*i,j_start:nb+j_start], 'b', label=f'True state ($U_{i+1}$)')
        plt.xlabel('$time$')
        plt.ylabel(f'$ \Re(U_{i+1})$')
        plt.legend()
    plt.savefig(SAVE + f"fig1enKF_{i}.png",format='png',dpi=400)
    # plt.figure()
    # plt.plot(time[0:nb],g[i,0:nb],label=f'inflation factor g for variable $U_{i//2}$')
    # plt.xlabel('time')
    # plt.ylabel('g')
    # plt.legend()
        #plt.savefig(SAVE + f"fig1enKF_{i}")


### plot state variables
# plt.figure()
# y_label=('$U_4$', '$U_5$', '$U_6$', '$U_7$')
# for i in range(4,8):
#     plt.subplot(2,2,i-4+1)
#     plt.plot(time[0:nb], Data_shell.T[2*i,0:nb], 'b')
#     if ((i==1) or (i==2)):
#         plt.plot(time[0:nb], y_obs[2*i,0:nb], '.k') 
#     plt.plot(time[0:nb], x_a_enkf[2*i,0:nb], 'r')
#     plt.fill_between(time[0:nb], x_a_enkf[i,0:nb] - 1.96*np.sqrt(P_a_enkf[i,i,0:nb]), x_a_enkf[i,0:nb] + 1.96*np.sqrt(P_a_enkf[i,i,0:nb]), facecolor='red', alpha=0.5)
#     plt.xlabel('Time', size=20)
#     plt.ylabel(y_label[i-4], size=20)
# plt.savefig(SAVE + "fig2enKF")
### compute Root Mean Squared Errors (RMSE) of the positions
print('RMSE(obs):', np.sqrt(np.mean((y_obs[:,0:nb] - Data_shell.T[range(2*(k_min_collocation-1),2*k_max_collocation,1),0:nb])**2))) ### A CACHER

print('RMSE(EnKF):', np.sqrt(np.mean((x_a_enkf[:,0:nb] - Data_shell.T[:,0:nb])**2,1))) 

plt.figure()
plt.semilogy([i for i in range(int(n/2))],np.mean(Data_shell.T[0::2,0:nb]**2 + Data_shell.T[1::2,0:nb]**2,1),label='Truth')
plt.semilogy([i for i in range(int(n/2))],np.mean(x_a_enkf[0::2,0:nb]**2 + x_a_enkf[1::2,0:nb]**2,1),label='pred')
plt.semilogy([i for i in range(1,int(n/2))],[i**(-1/3) for i in range(1,int(n/2))],'--',alpha=0.5)
plt.xlabel('shell number')
plt.ylabel('$log(<|U_n|^2>_T)$')
plt.legend()
plt.savefig(SAVE + "log_variance_enKF.png",format='png')

plt.figure()
plt.semilogy([i for i in range(int(n/2))],(np.mean((Data_shell.T[0::2,0:nb]**2 + Data_shell.T[1::2,0:nb]**2)**2 ,1))/np.mean((Data_shell.T[0::2,0:nb]**2 + Data_shell.T[1::2,0:nb]**2),1)**2,label='Truth')
plt.semilogy([i for i in range(int(n/2))],(np.mean((x_a_enkf[0::2,0:nb]**2 + x_a_enkf[1::2,0:nb]**2)**2 ,1))/np.mean((x_a_enkf[0::2,0:nb]**2 + x_a_enkf[1::2,0:nb]**2),1)**2,label='pred')
plt.xlabel('shell number')
plt.ylabel('$log(frac{<|U_n|^4>_T}{<|U_n|^2>_T})$')
plt.legend()
plt.savefig(SAVE + "log_kurtosis_enKF.png",format='png')

plt.figure()
plt.semilogy([i for i in range(int(n/2))], np.sqrt(np.mean((np.sqrt(x_a_enkf[0::2,0:nb]**2 + x_a_enkf[1::2,0:nb]**2) - np.sqrt(Data_shell.T[0::2,0:nb]**2 + Data_shell.T[1::2,0:nb]**2))**2,1))/np.mean(np.sqrt(Data_shell.T[0::2,0:nb]**2 + Data_shell.T[1::2,0:nb]**2)**2,1), marker='o')
plt.xlabel('shell number')
plt.ylabel('RMSE')
plt.savefig(SAVE + "RMSE_enKF.png",format='png')