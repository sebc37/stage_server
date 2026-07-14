import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



class GOY_PINN(nn.Module):

    def __init__(self,n_input,n_output,n_hidden,n_layers,batch_size,ic_size,n_fourier=256,sigma=1.0):
        super().__init__()
        # self.B_b = torch.randn(batch_size).to(device)
        # self.B_ic = torch.randn(ic_size).to(device)
        #self.norm =nn.BatchNorm1d(n_hidden)
        self.B_fourier = torch.randn(n_input, n_fourier).to(device) * sigma
        fourier_out_dim = 2 * n_fourier

        activation = nn.SiLU
        self.input_layer = nn.Sequential(*[
                                    nn.Linear(fourier_out_dim, n_hidden),
                                    activation()])#.to(device)

        self.hidden_layers = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(n_hidden, n_hidden),
                            activation()]) for _ in range(n_layers-1)])#.to(device)

        self.output_layer = nn.Linear(n_hidden, n_output)#.to(device)
        self.apply(init_weights)
        
    def fourier_embed(self, x):
        # x: (batch, n_input)
        # Projects to frequency space, then maps to [sin, cos] features
        x_proj = 2 * torch.pi * x @ self.B_fourier   # (batch, n_fourier)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (batch, 2*n_fourier)

    def forward(self,x):
        # self.B = torch.randn((x.shape)).to(device)
        # print("x :",x.shape[0])
        # # print(x)
        # intial_shape = x.shape
        # #x = x.squeeze()
        # if x.shape[0] == 10:
        #     x = torch.matmul(self.B_b,x)
        # else:
        #     x = torch.matmul(self.B_ic,x.T)

        
        # fourier_features = torch.cat([torch.cos(x),torch.sin(x)])
        # print('fourirer :' ,fourier_features.shape)
        # fourier_features = fourier_features.unsqueeze(-1)
        #x = self.norm(x)
        x = self.fourier_embed(x)
        x=self.input_layer(x)
        #x = self.norm(x)
        x=self.hidden_layers(x)
        x=self.output_layer(x)

        return x
    
################################################# DATASET ###############################################



# classe pour transformer les données en jeu de donnée des condittions initiales 
class initials_variables_data(Dataset):
    def __init__(self,X_ic,nbr_initial_t,k_min,k_max,t_0): # conditions initiales à t=0 et k sur l'ensemble des k
        
        #initialise the variables
        self.k_min = k_min
        self.k_max = k_max
        self.X_ic = torch.from_numpy(X_ic[self.k_min:self.k_max*2])  # select only the shells between k_min and k_max
        self.nbr_initial_t = nbr_initial_t
        
        self.x_initial = np.array([k for k in range(k_min,k_max*2)],dtype="float32") # shells selected for initial conditions
        self.t_initial = np.array([t_0 for _ in range(self.nbr_initial_t)],dtype="float32")  # 0 since the initial condition is defined for t=0
        
        # transform data and grid to the shape (k,t,u)
        self.tensor_data_ic = torch.tensor(self.X_ic, dtype=torch.float32)
        self.grille = np.meshgrid(self.x_initial,self.t_initial)
        self.grille = torch.tensor(self.grille, dtype=torch.float32).T.view(np.shape(self.x_initial)[0]*np.shape(self.t_initial)[0],2)
        
        self.tensor_data = torch.ones((np.shape(self.x_initial)[0]*np.shape(self.t_initial)[0],3), dtype=torch.float32)
        self.tensor_data[:,0:2] = self.grille
        self.tensor_data[:,2] = self.tensor_data_ic
        #print(f'shape tensor for initial conditions : {self.tensor_data.shape}')
    
    def __len__(self):
        #return the lenght of the dataset
        return self.x_initial.shape[0]*self.t_initial.shape[0]

    def __getitem__(self,idx):

        return self.tensor_data[idx,0], self.tensor_data[idx,1], self.tensor_data[idx,2]  #return the x and t values of the grid and value of X for the initial condition
    

# classe pour transformer les données en jeu de donnée des conditions de bord 
class boundary_variables_data(Dataset):
    def __init__(self,X_boundary,Npts,time,f,dt,mask):
        
        #N_fs = int(1/((f-0.1)*dt))
        self.mask = mask    
        self.X_boundary = torch.from_numpy(X_boundary[self.mask,:]) #[0:Npts,:]
        self.nb_k = np.shape(X_boundary)[1]
        self.nb_t = np.shape(X_boundary)[0]
       
        time = torch.arange(0,time,time/Npts,dtype=torch.float32) #(time/f)*1/N_fs  #10*(f-0.1)*dt
        shell = torch.arange(0,self.X_boundary.shape[1],1,dtype=torch.float32)
        grid_shell,grid_time = torch.meshgrid(shell,time[mask],indexing="xy")
        grid_shell = grid_shell.T.contiguous().view(len(self.mask)*self.X_boundary.shape[1],1) #Npts*self.X_boundary.shape[1]
        grid_time = grid_time.T.contiguous().view(len(self.mask)*self.X_boundary.shape[1],1) #Npts*self.X_boundary.shape[1]
        u_bc = torch.tensor(self.X_boundary).T.contiguous().view(len(self.mask)*self.X_boundary.shape[1],1) #Npts*self.X_boundary.shape[1]
        self.tensor_data_bc = torch.stack((grid_shell,grid_time,u_bc),1).view(len(self.mask)*self.X_boundary.shape[1],3) #Npts*self.X_boundary.shape[1]
        
        ############################### version boucle long ###############################
        # self.tensor_data_bc_bis = torch.ones((self.nb_k*self.nb_t,3), dtype=torch.float32) #columns: k, t, u
        # # trouver solution tq pour tout t, t!=0
        # for k in range(self.nb_k*self.nb_t): # data ordered as (k,t,u) in the grid
        #      self.tensor_data_bc_bis[k,0],self.tensor_data_bc_bis[k,1],self.tensor_data_bc_bis[k,2] = k//(self.nb_t),k%(self.nb_t),self.X_boundary[k%self.nb_t,k//self.nb_t]
        # print(f'shape tensor for boundary conditions: {self.tensor_data_bc.shape}')
        # t = torch.sub(self.tensor_data_bc_bis[:,2],self.tensor_data_bc[:,2])
        # print(t, torch.std_mean(t))
        # for k in range(self.nb_k):
        #     print(self.tensor_data_bc[k*(self.nb_t-3):k*(self.nb_t+3)]) 
        ###################################################################################
        
    def __len__(self):
        #return the lenght of the dataset
        #return self.nb_k*self.nb_t
        return self.tensor_data_bc.shape[0] 
    def __getitem__(self,idx):

        # return the element in that index (k,t,u)[idx] 
        return self.tensor_data_bc[idx,0],self.tensor_data_bc[idx,1],self.tensor_data_bc[idx,2]
    

# classe pour transformer les données en jeu de donnée des collocations points
class colocations_variables_data(Dataset):
    def __init__(self,X_Data_train):
        #initialise the variables

        self.X_Data_train = X_Data_train
        self.nb_colocation_pnt = len(self.X_Data_train[1])
        #colocation points aranged as (k,t,u) in the grid
        self.tensor_data_colocation = torch.tensor(self.X_Data_train, dtype=torch.float32).T.view(self.nb_colocation_pnt,3)
        #print(f'shape of tensor for collocation points :  {self.tensor_data_colocation.shape}')


    def __len__(self):
        #return the lenght of the dataset
        return self.nb_colocation_pnt
   

    def __getitem__(self,idx):

        # return the element in that index (k,t,u)[idx]
        return self.tensor_data_colocation[idx,0],self.tensor_data_colocation[idx,1]#,self.tensor_data_colocation[idx,2] #float(self.X_Data_train[idx][0]),float(self.X_Data_train[idx][1]) ,torch.tensor(self.X_Data_train[idx][2])   #return the x and t values of the grid and value of X for the physics loss
        #return self.x[j],self.t[i]  # This class only returns the x and t values of the grid not the velocity


class grid_data(Dataset): # créer la grille sur laquelle on veut inferer U(k,t) sous la forme (k,t)
    def __init__(self,k_min,k_max,t_min,t_max,Npts): # 
        #initialise the variables
        self.k_min = k_min
        self.k_max = k_max
        self.t_min = t_min
        self.t_max = t_max
        self.x = torch.arange(k_min,2*k_max,1,dtype=torch.float32)
        self.t = torch.arange(t_min,t_max,(t_max-t_min)/Npts,dtype=torch.float32)
        #self.t = torch.linspace(t_min, t_max, Npts, dtype=torch.float32)
        self.grid_k,self.grid_t = torch.meshgrid(self.x,self.t,indexing="xy")
        self.grid_k = self.grid_k.T.contiguous().view(Npts*(2*k_max-k_min),1)
        self.grid_t = self.grid_t.T.contiguous().view(Npts*(2*k_max-k_min),1)
        self.grid = torch.stack((self.grid_k,self.grid_t),1).view(Npts*(2*k_max-k_min),2)
        
        ############### version boucle long #################################
        #self.x = np.array([k for k in range(k_min,2*k_max)],dtype="float32")
        #self.t = np.arange(t_min,t_max,(t_max-t_min)/Npts,dtype="float32")
        #self.x, self.t = torch.tensor(self.x, dtype=torch.float32), torch.tensor(self.t, dtype=torch.float32)
        #self.grille = torch.ones((2*k_max-k_min)*Npts,2, dtype=torch.float32)
        # for k in range((2*k_max-k_min)*Npts):
        #     i = k%Npts #(t_max-t_min)
        #     j = k//Npts #(t_max-t_min)
        #     self.grille[k,0],self.grille[k,1] = self.x[j],self.t[i]
        # t = torch.sub(self.grille,self.grid)
        #######################################################################
        self.N_k = np.shape(self.x)[0] 
        self.N_t = np.shape(self.t)[0]
        #print(f'shape tensor of the grid : {self.grid.shape}')

    def __len__(self):
        #return the lenght of the dataset
        return self.N_t*self.N_k

    def __getitem__(self,idx):
        #print(f'idx {idx}')
        return self.grid[idx,0],self.grid[idx,1],idx  # This class only returns the x and t values of the grid not the velocity

class Lorenz_Dataset(Dataset):
    def __init__(self,path_data,dt,n):
        
        self.data = torch.tensor(np.load(path_data),dtype=torch.float32).H
        self.time = torch.arange(0,self.data.shape[0]*dt,dt,dtype=torch.float32)
        if(n!=1):
            self.slices = int(len(self.time)/n)
        else:
            self.slices =0 
        self.n = n

    def __len__(self):
        return(len(self.time))
    
    def __getitem__(self, idx):

        if idx + self.slices> self.__len__():
            idx = idx-1 - (self.slices - (self.__len__() - idx))
            point = self.data[idx:idx + self.slices,:]
        if self.slices==0:
            point = self.data[idx,:]
            return self.time[idx],point
        return self.time[idx: idx + self.slices],point

class Multilple_Lorenz(Dataset):
    def __init__(self,paths,dts,b_size):
        self.paths = paths
        self.dts=dts
        self.b_size = b_size
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        data = Lorenz_Dataset(self.paths[idx],self.dts[idx])
        
        return [DataLoader(data,batch_size=self.b_size,shuffle=True)]


# calcul des loss

class DynamicLossWeighter:
    def __init__(self, alpha=0.9,N=22,ic=1,bc=1,phy=1):
        """
        alpha: coefficient du moving average (ex: 0.9)
        """
        self.alpha = alpha
        # Initialisation des poids globaux
        self.lambda_ic  = 1/3
        self.lambda_bc  = 1/3
        self.lambda_r   = 1/3
       
        self.bc = bc
        self.ic = ic
        self.phy = phy  
        if self.phy:
            self.lmb_phy = [1/N for i in range(N)]
        else:
            self.lmb_phy = 1/3
            


    def compute_weights(self, loss_ic, loss_bc, loss_phy, model_params, epoch):
        """
        Calcule les nouveaux poids lambda selon les normes des gradients.
        
        loss_ic, loss_bc, loss_phy : tenseurs scalaires (non réduits)
        model_params : list(model.parameters())
        """

        def grad_norm(loss):
            """Calcule la norme L2 du gradient de `loss` par rapport aux paramètres."""
            grads = torch.autograd.grad(
                loss, model_params,
                retain_graph=True, create_graph=True, allow_unused=True
            )
            #print("shape : " , len(grads))
            total = sum(
                g.norm() ** 2
                for g in grads if g is not None
            )
            #for g in grads:
                #print("type of g : ", type(g))
                #print(torch.mean(g))
            return total.sqrt()
        
        #################################################################
        # calcul de la norme du gradient de la loss de chaque composante#
        #################################################################
        if self.bc:
            norm_bc = grad_norm(loss_bc)
        else:
            norm_bc = 0
        
        if self.ic:
            norm_ic = grad_norm(loss_ic)    
        else:
            norm_ic = 0

        # calcul de la norme pour chaque loss de physique
        if self.phy:
            grad_shell_list = []
            for l in loss_phy:
                norm_phy  = grad_norm(l)
                grad_shell_list.append(norm_phy)
            total =  norm_ic+ norm_bc + sum(grad_shell_list)
        else:

            norm_phy = grad_norm(loss_phy)
            total = norm_ic + norm_bc + norm_phy
            #grad_shell_list = [0 for i in range(len(loss_phy))]
        # print("norme ic : ", norm_ic, type(norm_ic))
        # print("norme phy : ",grad_shell_list, type(grad_shell_list))
        # print("norme bc : " , norm_bc, type(norm_bc))
        #print("norme bc : " , norm_bc, type(norm_bc))
        #print("norme phy : ",norm_r, type(norm_r))
        #norm_r #+norm_ic # dénominateur commun du numérateur
        #print("total : ", total, type(total)   )
        #################################################################
        # calcul des nouveaux facteurs de normalisation pour chaque loss#
        #################################################################
        if norm_ic != 0:
            lambda_ic_hat =  total / norm_ic # np.exp(-(10/(2*(epoch+1-10)))) * torch.log( +1)
        else:
            lambda_ic_hat = 0

        if norm_bc != 0:
            lambda_bc_hat =  total / norm_bc #np.exp(-(10/(2*(epoch+1-10)))) * torch.log( +1)
        else:
            lambda_bc_hat = 0

        #lambda_r_hat  = total / norm_r # il ya pas de lambda r quand on normalise pour toutes les shells ()
        if self.phy:
            lmbphy_hat_list = [] # tous les nouveaux lambdas physiques
            for lmb in grad_shell_list:
                lmbphy_hat_list.append( total / lmb ) #np.exp(-(10/(2*(epoch+1-10)))) * torch.log( +1)
        else:
            lmbphy_hat_list = total / norm_phy  #np.exp(-(10/(2*(epoch+1-10)))) * torch.log( +1)
        # print("lmb ic pondéré : ",lambda_ic_hat,type(lambda_ic_hat))
        # print("lmb bc pondéré : ",lambda_bc_hat,type(lambda_bc_hat))
        # print("lmb phy pondéré : ",lmbphy_hat_list, type(lmbphy_hat_list))
        #print("lmb bc pondéré : ",lambda_bc_hat,type(lambda_bc_hat))
        #print("lmb phy pondéré : ",lambda_r_hat,type(lambda_r_hat))
        return  lambda_ic_hat,lambda_bc_hat ,lmbphy_hat_list #lambda_r_hat

    def update(self, loss_ic, loss_bc, loss_r, model_params,epoch):
        """
        Met à jour les poids avec le moving average :
            lambda_new = alpha * lambda_old + (1 - alpha) * lambda_hat_new
        """
        with torch.no_grad():
            l_ic, l_bc, lmbphy_list = self.compute_weights(
                loss_ic, loss_bc, loss_r, model_params,
            epoch)
            
            self.lambda_ic = self.alpha * self.lambda_ic + (1 - self.alpha) * l_ic#.item()
            self.lambda_bc = self.alpha * self.lambda_bc + (1 - self.alpha) * l_bc#.item()
            #self.lambda_r  = self.alpha * self.lambda_r  + (1 - self.alpha) * l_r#.item()
            # Mettre à jour les poids physiques
            if self.phy:
                for i, lmb in enumerate(lmbphy_list):
                    self.lmb_phy[i] = self.alpha * self.lmb_phy[i] + (1 - self.alpha) * lmb#.item()
            else:
                self.lmb_phy = self.alpha * self.lmb_phy + (1 - self.alpha) * lmbphy_list#.item()


    def weighted_loss(self, loss_ic, loss_bc, loss_r, epoch):
        """Retourne la loss totale pondérée."""
        #print("lmb used bc : ",self.lambda_bc,type(self.lambda_bc))
        #print("lmb used phy : ",self.lambda_r,type(self.lambda_r))
        if self.phy:
            return (
                self.lambda_ic * loss_ic +
                self.lambda_bc * loss_bc +
                + sum(np.exp(-(10/(2*(epoch+1-10)))) * l * r for l, r in zip(self.lmb_phy, loss_r))
            )
        else:
            return (
                self.lambda_ic * loss_ic +
                self.lambda_bc * loss_bc +
                self.lmb_phy * loss_r
            )