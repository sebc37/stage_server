import os
import numpy as np
from custom_sampler import *
from architecture import *
from parser import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.func import functional_call, vmap, jacrev
import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse


class Train_PINN():

    def __init__(self,learning_rate,nbr_iteration,w_1,w_2,w_3,w_4,sample_phy,iteration=True,epoch=1000,physic=True,collocation=True,initial=True,normalize_phy=True,inline_phy=True,enrgie_normalization=True):
        self.learning_rate = learning_rate
        self.nbr_iteration = nbr_iteration
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.w_4 = w_4
        self.normalize_enrgie = enrgie_normalization
        self.optimizer = torch.optim.Adam(model.parameters(),lr = self.learning_rate)

        self.warmup  = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, total_iters=1000)
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=9000,eta_min=1e-6)
        self.scheduler_bis = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.warmup, self.cosine], milestones=[1000])
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
        self.optimizer_lbfgs = optim.LBFGS(
    model.parameters(),
    lr=0.1,
    max_iter=20,
    history_size=50,
    line_search_fn='strong_wolfe'  # crucial pour la stabilité
)
        self.iteration = iteration
        self.epoch = epoch
        self.physic = physic
        self.collocation = collocation
        self.initial = initial
        self.normalize_phy = normalize_phy
        self.inline_phy = inline_phy
        self.sample_phy = sample_phy

    def train(self):

        loss = np.zeros((self.nbr_iteration,1))
        loss_physics_tracker = np.zeros((self.nbr_iteration,1))
        loss_colocation_tracker = np.zeros((self.nbr_iteration,1))
        loss_boundary_conditions_tracker = np.zeros((self.nbr_iteration,1))
        loss_initial_conditions_tracker = np.zeros((self.nbr_iteration,1))
        lmb_tracker_phy = np.zeros((self.nbr_iteration,1))
        lmb_tracker_bc = np.zeros((self.nbr_iteration,1))
        loss_trackeur_phy = np.zeros((22,self.nbr_iteration))
        lmb_phy_trackeur = np.zeros((22,self.nbr_iteration))
        lmb_ic_trackeur = np.zeros((self.nbr_iteration,1))
        weighter = DynamicLossWeighter(alpha=0.9,N=22)

        

        params = list(model.parameters())

        def ic_losses():
            initial_train_data = initial_train_dataset.tensor_data.to(device) #torch.tensor(initial_rows, dtype=torch.float32)
            u_pd_ini = model(initial_train_data[:, 0:2]).to(device)
            u_exa_ini = initial_train_data[:,2:3]
            loss_initital_conditions = self.w_1*torch.mean((u_pd_ini-u_exa_ini)**2).to(device)
            return loss_initital_conditions
        
        def bc_losses():
            boundary_train_data = boundary_train_dataset.tensor_data_bc.to(device)#torch.tensor(boundary_rows, dtype=torch.float32)
            u_pd_bou = model(boundary_train_data[:, 0:2]).to(device)
            u_exa_bou = boundary_train_data[:,2:3]
            loss_bc = self.w_2*torch.mean((u_pd_bou-u_exa_bou)**2).to(device)
            return loss_bc

        def colocation_losses():
            colocation_train_data = colocation_dataset.tensor_data_colocation#.to(device)
            loss_colocation = 0
            for i in range(360): 
                vect = colocation_train_data[i*3003: (i+1)*3003,0:2].to(device)
                u_pd_colocation = model(vect).to(device)
            
            
                loss_colocation += self.w_3*torch.mean((u_pd_colocation[:]-vect)**2).to(device)

            loss_colocation = loss_colocation/(360)
            return loss_colocation
        
        def physics_losses():
            grid_train_data = grid_dataset.grid.requires_grad_(True).to(device)
            u_pd = model(grid_train_data).to(device)
            u_t = torch.autograd.grad(u_pd, grid_train_data, torch.ones_like(u_pd), create_graph=True)[0][:,1:2].to(device)
            # calcul de la physique
            u_pd = u_pd.view(2*k_max-k_min,Npts).T #Npts
            u_t = u_t.view(2*k_max-k_min,Npts).T
            u_t_phy = u_t[self.sample_phy,:]
            u_pd_phy = u_pd[self.sample_phy,:]
            u_pd_im = u_pd_phy[:,1::2]
            u_pd_real = u_pd_phy[:,::2]
            u_t_im = u_t_phy[:,1::2]
            u_t_real = u_t_phy[:,::2]
            # print(u_pd_im.shape)
            # print(u_t_im.shape)
            if self.normalize_enrgie:
                a=[(1/(U0*K[i]**(-2/3))) for i in range(k_max)]
                print(a)
            else:
                a=[1 for i in range(k_max)]
            # on veut calculer la loss physique sur le shells où il y a des collocations points
            GOY_physics_im = torch.zeros(len(self.sample_phy),k_max).to(device)
            GOY_physics_real = torch.zeros(len(self.sample_phy),k_max).to(device)

            # calcul sur les premiers modes
            GOY_physics_im[:,0] = a[0]*(u_t_im[:,0] - K[0]*(u_pd_real[:,1]*u_pd_real[:,2] - u_pd_im[:,1]*u_pd_im[:,2]) 
            + nu*(K[0]**2)*u_pd_im[:,0])
            
            GOY_physics_im[:,1] = a[1]*(u_t_im[:,1] -K[1]*(u_pd_real[:,2]*u_pd_real[:,3] - u_pd_im[:,2]*u_pd_im[:,3])
            +(eps/lmb)*K[1]*(u_pd_real[:,0]*u_pd_real[:,2] - u_pd_im[:,0]*u_pd_im[:,2])
            + nu*(K[1]**2)*u_pd_im[:,1])

            GOY_physics_real[:,0] = a[0]*(u_t_real[:,0] - K[0]*(u_pd_real[:,1]*u_pd_im[:,2] - u_pd_im[:,1]*u_pd_real[:,2]) 
            + nu*(K[0]**2)*u_pd_real[:,0])
            
            GOY_physics_real[:,1] = a[1]*(u_t_real[:,1] -K[1]*(u_pd_real[:,2]*u_pd_im[:,3] - u_pd_im[:,2]*u_pd_real[:,3])
            +(eps/lmb)*K[1]*(u_pd_real[:,0]*u_pd_im[:,2] - u_pd_im[:,0]*u_pd_real[:,2])
            + nu*(K[1]**2)*u_pd_real[:,1])


            # cacul à l'intétrieur du domaine
            for i in range (2,k_max-2):
                GOY_physics_im[:,i] = a[i]*(u_t_im[:,i] 
                - K[i]*(u_pd_real[:,i+1]*u_pd_real[:,i+2] - u_pd_im[:,i+2]*u_pd_im[:,i+1])
                +(eps/lmb)*K[i]*(u_pd_real[:,i-1]*u_pd_real[:,i+1] - u_pd_im[:,i-1]*u_pd_im[:,i+1])
                -((eps-1)/(lmb**2))*K[i]*(u_pd_real[:,i-2]*u_pd_real[:,i-1] - u_pd_im[:,i-2]*u_pd_im[:,i-1])
                +nu*(K[i]**2)*u_pd_im[:,i])

                GOY_physics_real[:,i] = a[i]*(u_t_real[:,i] 
                - K[i]*(u_pd_real[:,i+1]*u_pd_im[:,i+2] - u_pd_real[:,i+2]*u_pd_im[:,i+1])
                +(eps/lmb)*K[i]*(u_pd_real[:,i-1]*u_pd_im[:,i+1] - u_pd_im[:,i-1]*u_pd_real[:,i+1])
                -((eps-1)/(lmb**2))*K[i]*(u_pd_real[:,i-2]*u_pd_im[:,i-1] - u_pd_im[:,i-2]*u_pd_real[:,i-1])
                +nu*(K[i]**2)*u_pd_real[:,i])

            # calcul sur les derniers modes
            GOY_physics_im[:,k_max-2] = a[k_max-2]*(u_t_im[:,k_max-2] 
            + K[k_max-2]*(eps/lmb)*(u_pd_real[:,k_max-3]*u_pd_real[:,k_max-1]-u_pd_im[:,k_max-3]*u_pd_im[:,k_max-1])
            - K[k_max-2]*((eps-1)/(lmb**2))*(u_pd_real[:,k_max-4]*u_pd_real[:,k_max-3] - u_pd_im[:,k_max-4]*u_pd_im[:,k_max-3])
            + nu*(K[k_max-2]**2)*u_pd_im[:,k_max-2])
            
            GOY_physics_im[:,k_max-1] = a[k_max-1]*(u_t_im[:,k_max-1]
            -((eps-1)/(lmb**2))*K[k_max-1]*(u_pd_real[:,k_max-3]*u_pd_real[:,k_max-2] - u_pd_im[:,k_max-3]*u_pd_im[:,k_max-2])
            + nu*(K[k_max-1]**2)*u_pd_im[:,k_max-1])

            GOY_physics_real[:,k_max-2] = a[k_max-2]*(u_t_real[:,k_max-2] 
            + K[k_max-2]*(eps/lmb)*(u_pd_real[:,k_max-3]*u_pd_im[:,k_max-1]-u_pd_im[:,k_max-3]*u_pd_real[:,k_max-1])
            - K[k_max-2]*((eps-1)/(lmb**2))*(u_pd_real[:,k_max-4]*u_pd_im[:,k_max-3] - u_pd_im[:,k_max-4]*u_pd_real[:,k_max-3])
            + nu*(K[k_max-2]**2)*u_pd_real[:,k_max-2])
            
            GOY_physics_real[:,k_max-1] = a[k_max-1]*(u_t_real[:,k_max-1]
            -((eps-1)/(lmb**2))*K[k_max-1]*(u_pd_real[:,k_max-3]*u_pd_im[:,k_max-2] - u_pd_im[:,k_max-3]*u_pd_real[:,k_max-2])
            + nu*(K[k_max-1]**2)*u_pd_real[:,k_max-1])
            # à compléter
            
            return GOY_physics_real,GOY_physics_im
        
        def total_losses_weithed():
            if self.initial:
                loss_initial_conditons = ic_losses()
            
            loss_boundary_conditions = bc_losses()
            if self.physic:
                GOY_physics_real, GOY_physics_im = physics_losses()
            list_loss_phy = []
                    
            for i in range(len(weighter.lmb_phy)):
                list_loss_phy.append(weighter.lmb_phy[i] *torch.mean ((GOY_physics_real[:,i]**2 + GOY_physics_im[:,i]**2)).to(device) )
                        #u_t[:,i] -K[i]*u_pd[:,i+1]*u_pd[:,i+2] +K[i]*eps/lmb*u_pd[:,i-1]*u_pd[:,i+1] + K[i]*((eps-1)/lmb**2)*u_pd[:,i-2]*u_pd[:,i-1] + nu*K[i]*K[i]*u_pd[:,i] # à confirmer
            loss_physics =  sum(list_loss_phy) #self.w_4*torch.mean(sum(list_loss_phy)).to(device) #GOY_physics_real**2+GOY_physics_im**2).to(device)


            if iteration % 100 == 0 and iteration<=10000:
                weighter.update( loss_ic=loss_initial_conditons,loss_bc=loss_boundary_conditions, loss_r=list_loss_phy, model_params=params)
            total_loss = weighter.weighted_loss(loss_ic=loss_initial_conditons,loss_bc=loss_boundary_conditions, loss_r=list_loss_phy)


            loss[iteration]=total_loss.cpu().detach().numpy()
            if self.physic:
                loss_physics_tracker[iteration] = loss_physics.cpu().detach().numpy()
                loss_trackeur_phy[:,iteration] = [l.cpu().detach().numpy() for l in list_loss_phy]
            else:
                loss_physics_tracker[iteration] = loss_physics
            if self.collocation:
                loss_colocation_tracker[iteration] = loss_colocation.cpu().detach().numpy()
            else:
                loss_colocation_tracker[iteration] = 0
            
            loss_boundary_conditions_tracker[iteration] = loss_boundary_conditions.cpu().detach().numpy()
            lmb_tracker_bc[iteration] = weighter.lambda_bc.cpu().detach().numpy()
            if self.initial:
                loss_initial_conditions_tracker[iteration] = loss_initial_conditons.cpu().detach().numpy()
                lmb_ic_trackeur[iteration] = weighter.lambda_ic.cpu().detach().numpy()
            else:
                loss_initial_conditions_tracker[iteration] = 0
            #lmb_tracker_phy[iteration] = weighter.lambda_r#.cpu().detach().numpy()
            
            for i in range(22):
                lmb_phy_trackeur[i,iteration] = weighter.lmb_phy[i].cpu().detach().numpy()

            return total_loss 
 
        if self.iteration:

            def closure():
                    self.optimizer_lbfgs.zero_grad()
                    total_loss = total_losses_weithed() 
                    total_loss.backward()
                    return total_loss

            for iteration in tqdm.tqdm(range(self.nbr_iteration)):
                
                if iteration <=10000:
                    #print('optimizer : Adam')
                    self.optimizer.zero_grad()
                    total_loss = total_losses_weithed()
                    total_loss.backward()
                    self.optimizer.step()
                    self.scheduler_bis.step()
                else: 
                    #print('optimizer : LBFGS')
                    self.optimizer_lbfgs.step(closure)

                
        
               

        else:
                # regarder comment calculer les loss 
                loss_func = torch.nn.MSELoss()
                for ep in range(0,self.epoch):
                    loss_boundary_conditions = 0
                    loss_initital_conditions = 0
                    loss_colocation = 0
                    loss_physics = 0
                    for idx,ic in enumerate(Dataloader_ic):
                        # ic[0].to(device)
                        # ic[1].to(device)
                        ic_pred = torch.stack((ic[0],ic[1])).T.to(device)
                        ic_pred.requires_grad_()
                        u_pd_ini = model(ic_pred)
                        u_exa_ini = ic[2].to(device)
                        loss_initital_conditions = loss_initital_conditions + self.w_1*loss_func(u_pd_ini,u_exa_ini)

                        
                    for idx,bc in enumerate(Dataloader_bc):
                        # bc[0].to(device)
                        # bc[1].to(device)
                        bc_pred = torch.stack((bc[0],bc[1])).T.to(device)
                        u_pd_bc = model(bc_pred)
                        u_exa_bc = bc[2].to(device)
                        loss_boundary_conditions = loss_boundary_conditions +  self.w_2*loss_func(u_pd_bc,u_exa_bc)

                    for idx,cl in enumerate(Dataloader_cl):
                        cl_pred = torch.stack((cl[0],cl[1])).T.to(device)
                        cl_pred.requires_grad_()
                        u_pd_cl = model(cl_pred)
                        u_exa_cl = cl[2].to(device)
                        loss_colocation = loss_colocation +  self.w_3*loss_func(u_pd_cl,u_exa_cl)

                    for idx,grid in enumerate(Dataloader_grid):
                        grid_pred = torch.stack((grid[0],grid[1])).T.to(device)
                        grid_pred.requires_grad_()
                        u_pd = model(grid_pred)
                        du_dt = torch.autograd.grad(u_pd, grid_pred, torch.ones_like(u_pd), create_graph=True)[0][:,1:2]
                        m = Dataloader_grid.batch_sampler.m
                        k_min_phy = Dataloader_grid.batch_sampler.k_min_grid
                        k_max_phy = Dataloader_grid.batch_sampler.k_max_grid-2
                        nb_k = Dataloader_grid.batch_sampler.k_max_grid - Dataloader_grid.batch_sampler.k_min_grid

                        u_pd = u_pd.view(nb_k,m).T
                        du_dt = du_dt.view(nb_k,m).T

                        GOY_physics = torch.zeros(m,(k_max_phy-2 - k_min_phy)).to(device)
                        for i in range(k_min_phy,k_max_phy-2):
                             GOY_physics[:,i-k_min_phy] = du_dt[:,i] -K[i]*u_pd[:,i+1]*u_pd[:,i+2] +K[i]*eps/lmb*u_pd[:,i-1]*u_pd[:,i+1] + K[i]*((eps-1)/lmb**2)*u_pd[:,i-2]*u_pd[:,i-1] + nu*K[i]*K[i]*u_pd[:,i]
                            
                        loss_physics = loss_physics + self.w_4*torch.mean(GOY_physics**2)
        # print(iteration)
        # print(total_loss)
            
            
        return {
    "loss": loss,
    "loss_physics_tracker": loss_physics_tracker,
    "loss_colocation_tracker": loss_colocation_tracker,
    "loss_boundary_conditions_tracker": loss_boundary_conditions_tracker,
    "loss_initial_conditions_tracker": loss_initial_conditions_tracker,
    "lmb_tracker_bc": lmb_tracker_bc,
    "lmb_tracker_phy": lmb_tracker_phy,
    "u_t": None,
    "loss_trackeur_phy": loss_trackeur_phy,
    "lmb_phy_trackeur": lmb_phy_trackeur,
    "lmb_ic_trackeur": lmb_ic_trackeur,
}


def get_ntk(model, x1, x2):
    """
    Computes the NTK matrix K[i,j] = <J(x1[i]), J(x2[j])>
    Args:
        model : nn.Module
        x1    : (N, d) tensor
        x2    : (M, d) tensor
    Returns:
        K     : (N, M) NTK matrix
    """
    params = dict(model.named_parameters())

    def fnet_single(params, x):
        # Forward pass for a single input, returns shape (out_dim,)
        return functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)

    # Jacobian: (N, out_dim, num_params_flat) — via vmap over the batch
    def compute_jacobian(x):
        # jacrev returns a dict of per-param Jacobians; we flatten and concat
        jac = jacrev(fnet_single)(params, x)  # dict of tensors
        # Flatten all param Jacobians into a single vector per output dim
        jac_flat = torch.cat([j.flatten(1) for j in jac.values()], dim=1)
        return jac_flat  # (out_dim, P)

    J1 = vmap(compute_jacobian)(x1)  # (N, out_dim, P)
    J2 = vmap(compute_jacobian)(x2)  # (M, out_dim, P)

    # NTK: K[i,j] = sum over output dims and params: J1[i] @ J2[j]^T
    # Einsum: N x out x P, M x out x P -> N x M
    K = torch.einsum('nop,mop->nm', J1, J2)
    return K

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
                X_filtered[i,j] = X_subset[i,j]
                X_posx.append(j)
                X_posy.append(i/nb_line)
                X_value.append(X_subset[i,j])#-mean_mode[j])/std_mode[j]) # centré réduit

            else:
                X_filtered[i,j] = None
    
    pourcentage_filtered = nb_line*ratio/nb_line*100
    X_dataset.append(X_posx)
    X_dataset.append(X_posy)
    X_dataset.append(X_value)
    return X_filtered,X_dataset,mean_mode,var_mode,std_mode,pourcentage_filtered


def reduced_center(X,mean,std):
    for k in range(X.shape[1]):
        X[:,k]= (X[:,k]-mean[k])/std[k]
    return X



def test_loss(Data_train,grille_datatset):

    # grid_train_data = grille_datatset.grille # donne la grille des (k,t)
    # grid_train_data.requires_grad = True # permet d'appliqué grad
    data = Data_train[:,2*k_min_collocation:2*k_max]
    data_real = data[:,::2]
    data_im = data[:,1::2]
    grille = [t for t in range(Npts)]
    #u = torch.tensor(Data_train[:,k_min:2*k_max],requires_grad=True) #tensor des U(k,t)  #view(Npts,k_max-k_min)
    #tensor_split = torch.split(tensor_data,k_max-k_min,1)
    u_t = np.copy(Data_train[:,2*k_min_collocation:2*k_max-2])
    for k in range(2*k_min_collocation,2*k_max-2):
        for t in range(0,Npts):
            #u_t[t,k-k_min_collocation] = (data[t+1,k-k_min_collocation]*np.exp(nu*K[k]*dt)-data[t,k-k_min_collocation])/dt
            if t!=0 and t!=Npts-1:
                u_t[t,k-2*k_min_collocation] = (data[t+1,k-2*k_min_collocation]-data[t-1,k-2*k_min_collocation])/(2*(dt))
            if t==0:
                u_t[t,k-2*k_min_collocation] = (data[t+1,k-2*k_min_collocation]-data[t,k-2*k_min_collocation])/(dt)
            if t==Npts-1:
                u_t[t,k-2*k_min_collocation] = (data[t,k-2*k_min_collocation]-data[t-1,k-2*k_min_collocation])/(dt)

    u_t_real = u_t[:,::2]
    u_t_im = u_t[:,1::2]

    u_t_im = torch.tensor(u_t_im)
    u_t_real = torch.tensor(u_t_real)

    GOY_physics_real = torch.zeros(Npts,(k_max-2 - k_min_collocation))
    GOY_physics_im = torch.zeros(Npts,(k_max-2 - k_min_collocation))

    
    for i in range (k_min_collocation,k_max-2):
            
            j= i-k_min_collocation

            GOY_physics_real[:,j] = u_t_real[:,j] 

            -K[i]*(data_real[:,j+1]*data_im[:,j+2] + data_im[:,j+1]*data_real[:,j+2]) 

            +K[i]*(eps/lmb)*(data_real[:,j-1]*data_im[:,j+1]+data_im[:,j-1]*data_real[:,j+1]) 
            
            - K[i]*((eps-1)/lmb**2)*(data_real[:,j-2]*data_im[:,j-1] + data_im[:,j-2]*data_real[:,j-1]) 

            + nu*K[i]*K[i]*data_real[:,j] 

            GOY_physics_im[:,j] = u_t_im[:,j] 

            -K[i]*(data_real[:,j+1]*data_real[:,j+2] - data_im[:,j+1]*data_im[:,j+2]) 

            +K[i]*(eps/lmb)*(data_real[:,j-1]*data_real[:,j+1] - data_im[:,j-1]*data_im[:,j+1]) 
            
            - K[i]*((eps-1)/lmb**2)*(data_real[:,j-2]*data_real[:,j-1] - data_im[:,j-2]*data_im[:,j-1]) 

            + nu*K[i]*K[i]*data_im[:,j]
            plt.figure()
            plt.plot(grille,GOY_physics_real[:,j].detach().numpy()**2,label=f'Loss shell{i}')
            plt.legend()
            plt.savefig(PATH + f"loss_shell_real{i}")

            plt.figure()
            plt.plot(grille,GOY_physics_im[:,j].detach().numpy()**2,label=f'Loss shell{i}')
            plt.legend()
            plt.savefig(PATH + f"loss_shell_im{i}")

    print(torch.max(GOY_physics_real**2),torch.min(GOY_physics_real**2),torch.std_mean(GOY_physics_real**2))
    print(torch.max(GOY_physics_im**2),torch.min(GOY_physics_im**2),torch.std_mean(GOY_physics_im**2))
    loss_physics = (torch.mean(GOY_physics_im**2) +torch.mean(GOY_physics_real**2))/2
    return loss_physics,GOY_physics_real,GOY_physics_im

#########################################################################
#               Paramètres Pinn et entrainement                         #
#########################################################################
parser = argparse.ArgumentParser()
parser.add_argument("config", help="chemin vers YAML avec config du PINN",type=str)
args = parser.parse_args().config

config = parse_config(args)
print(config)
##### changer l'ordre  des k ==> k1,t0,k2,t0...kn,t0;k1,t1,k2,t1...kn,t1 etc#########


# check if cuda available ==> log


command = torch.cuda.is_available()
print(f'cuda is available : {command}')

######### The data generated by the shell model #######
PATH = config["path_graph"]  #r"/home/s26calme/Documents/code_stage/GOY-main/" # r"/home/s26calme/Documents/code_stage/Donnees/GOY_modele/Parametre_Ewen/"
path_data = config["path_data"]#PATH + "data.dat"

data =  np.loadtxt(path_data,dtype=np.float32) # charge le jeu de données
Nmax = np.shape(data)[0] # nombres de pas de temps
debut = int(0.1*Nmax) # skip la phase de stabilisation

Data_shell = data[debut:debut+20000,:] # on garde que la partie réelle de chaque shell
Npts = np.shape(Data_shell)[0] # nombre de pas dans le temps

# nb of shells selected for training the PINN on collocatin point
k_min_collocation = config["k_min_collocation"]#4 
k_max_collocation = config["k_max_collocation"]#10 

#nb of shells for training on boundary conditions
k_bc_min = config["k_min_boundary"]#0
k_bc_max = config["k_max_boundary"]#4 

ratio = config["ratio"]
nb_couche = config["PINN"][1]
largeur_couche = config["PINN"][0]
nbr_iteration = config["nb_iter"]
physic = config["physic"]
initial = config["initial"]
collocation = config["collocation"]
normalize_phy = config["normalize_phy"]
inline_phy = config["inline_phy"]
ratio_sample_obs = config["ratio_sample_obs"]
ratio_sample_phy = config["ratio_sample_phy"]

# retourne un dataset pour plot , var,std,et mean pour chaque mode et les colocation point centré réduit
Data_filtered, Data_train, mean, Var_mode, Std_mode, perc, = filter_mode(Data_shell,2*k_min_collocation,2*k_max_collocation,0,ratio,123456)


#Data_shell = reduced_center(Data_shell,mean,Std_mode) # centré réduit tous les modes 
Data_ic = Data_shell[0,:] # prends tous le spoints en t=0
random_samples = np.random.choice(Npts, size=int(ratio_sample_obs*Npts), replace=False) # prends 18000 points de la partie filtrée pour les collocation points
Data_bc = Data_shell[:,k_bc_min:2*k_bc_max] # prends tous les points de bords (shell allant de 0->3 avec 3 shell de forçage)
print(Data_bc.shape)
print("Moyenne de l'ensemble des mode",np.mean(Data_shell))
print("std de tous les modes ", np.std(Data_shell))

random_samples_phy = np.random.choice(Npts, size=int(ratio_sample_phy*Npts), replace=False)

#########  carateristics of the shell model ############

k0 = 0.125    # largest scale
lmb = 2.0     #ratio between consecutive scales
eps =  0.5     # for the NL coefficients
nu = 1.e-7     # vicosité
nb_shell = 22 
dt = 8.9999e-5
f=99999.9 # sauvegarde tous les f points
time = 1.0
Steps = time/dt # nombre de pas
N_fs = int(1/((f-0.1)*dt)) # enregistrement tous les N_fs pas 
############ parameters for the PINN ############


U0 = np.mean(Data_shell[:,0]**2 + Data_shell[:,1]**2)
# k_min et k_max sur l'ensemble des shells à reconstituter 
k_min = min(k_min_collocation,k_bc_min)
k_max = max(k_bc_max,k_max_collocation)

# list of coefficient kn for all shells
K = [k0*lmb**i for i in range(k_min,k_max)]



############ initialization of model ###########
torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GOY_PINN(n_input=2,n_output=1,n_hidden=largeur_couche,n_layers=nb_couche,batch_size=1,ic_size=1)
model.to(device)


nbr_initial_t = 1     # Only t=0 for defining the initial condition
t_min = 0.1*time    # t initial pour la grille          
t_max = time # tmax pour la grille 


############# initialization of dataset for training #################

# sample m points consécutifs à un tps au hasard sur les shells pour calculer loss physic 

#point_grille = Npts-debut
initial_train_dataset = initials_variables_data(Data_ic,nbr_initial_t,k_min,k_max,t_0=t_min)
boundary_train_dataset = boundary_variables_data(X_boundary=Data_bc,Npts=Npts,time=time,f=f,dt=dt,mask=random_samples)
colocation_dataset = colocations_variables_data(Data_train)
grid_dataset = grid_data(k_min,k_max,t_min,t_max,Npts=Npts)

#grid_test = grid_data(0,22,t_min,t_max,Npts=Npts)

batch_size_bc = int(0.01*Npts)
batch_size_cl = int(0.1*colocation_dataset.nb_colocation_pnt)
batch_size_grid = int(0.01*(k_max*Npts))

if collocation:
    sampler_grid = SamplerOverGrid(100,k_min_collocation-2,k_max,Npts=Npts) 
    #batch_sampler_grid = BatchSampler(sampler=sampler_grid,batch_size=batch_size_grid,drop_last=False)

    Dataloader_ic = DataLoader(initial_train_dataset,batch_size=10,drop_last=False)
    Dataloader_bc = DataLoader(boundary_train_dataset,batch_size=batch_size_bc,shuffle=True,drop_last=False)
    Dataloader_cl = DataLoader(colocation_dataset,batch_size=batch_size_cl,shuffle=True,drop_last=False)
    Dataloader_grid = DataLoader(grid_dataset,batch_sampler=sampler_grid)


# for idx,data in enumerate(Dataloader_grid):
#     print(data)
#     print(idx)



# boundary_train_dataset.tensor_data_bc.to(device)
# colocation_dataset.tensor_data_colocation.to(device)
# grid_dataset.grille.to(device)

# quick integration check using the RK4 helper
#solution = RK4(Data_shell[0,2:10], 0, 8, 1, dt=1.0e-5, K=K, eps=eps, lmb=lmb, nu=nu)
#print(solution[-1,])
#print("RK4 returned array of shape", solution.shape)

#test_loss(Data_train=Data_shell,grille_datatset=grid_dataset)

############################################################################################
#             Entrainement du PINN et évaluation de la performance                         #
############################################################################################

learning_rate,nbr_iteration,w_1,w_2,w3,w_4 = 0.001,nbr_iteration,1,1,1,1
t = Train_PINN(learning_rate,nbr_iteration,w_1,w_2,w3,w_4,sample_phy=random_samples_phy,
               physic=physic,initial=initial,collocation=collocation,
               normalize_phy= normalize_phy,inline_phy=inline_phy)
Total_loss = t.train()
model.eval().to(device)

U = model(grid_dataset.grid.to(device))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of parameters: {total_params}')
U_split = torch.split(U,Npts)
U = torch.cat(tuple(k for k in U_split),1).to(device)
U = U.cpu().detach().numpy()#.detach().numpy().reshape(t_max-t_min,k_max-k_min)
U_exa = Data_shell[0:Npts,k_min:2*k_max]

square_error = ((np.sqrt(U[:,0::2]**2 + U[:,1::2]**2) - np.sqrt(U_exa[:,0::2]**2 + U_exa[:,1::2]**2))**2)
amp = np.sqrt(U[0:Npts,0::2]**2 + U[0:Npts,1::2]**2) 
amp_exa = np.sqrt(U_exa[0:Npts,0::2]**2 + U_exa[0:Npts,1::2]**2)  
rmse = np.sqrt(np.mean((amp - amp_exa)**2,0))/np.mean(amp_exa,0)   
#rmse = np.sqrt(np.mean(square_error,0))/(np.mean(np.sqrt(U_exa[:,0::2]**2 + U_exa[:,1::2]**2),0))
print("RMSE:", rmse)
plt.figure()
plt.semilogy(rmse,label='RMSE',marker='o',linestyle='None')
plt.xlabel('numéro de couche')
plt.ylabel('RMSE')
plt.savefig(PATH + f"RMSE.png") #_{int(ratio*100)}
plt.close()

phys = Total_loss["loss_trackeur_phy"]
# plt.figure()
# for i in range(22):
#     plt.semilogy(phys[i,:],label='Loss physique shell'+str(i))
# plt.xlabel('Iterations')
# plt.ylabel('Loss physique')
# plt.legend()
# plt.savefig(PATH + f"loss_phy_shells.png")
# plt.close()

n_shells = 22
shells_per_fig = 3
n_figs = (n_shells + shells_per_fig - 1) // shells_per_fig  # ceil division

colors = cm.viridis(np.linspace(0, 1, n_shells))

fig, ax = plt.subplots()
for i in range(n_shells):
    label = f'Shell {i}' if i in (0, n_shells // 2, n_shells - 1) else None
    ax.semilogy(phys[i, :], color=colors[i], label=label)

ax.set_xlabel('Iterations')
ax.set_ylabel('Loss physique')
ax.set_title('Loss physique — all shells')
ax.legend()
fig.savefig(PATH + "loss_phy_shells_all.png")
plt.close(fig)



lmbs_shell = Total_loss["lmb_phy_trackeur"]

colors = cm.viridis(np.linspace(0, 1, n_shells))

fig, ax = plt.subplots()
for i in range(n_shells):
    label = f'Shell {i}' if i in (0, n_shells // 2, n_shells - 1) else None
    ax.semilogy(lmbs_shell[i, :], color=colors[i], label=label)

ax.set_xlabel('Iterations')
ax.set_ylabel('Lambda')
ax.set_title('Lambda — all shells')
ax.legend()

fig.savefig(PATH + "lambda_shells_all.png")
plt.close(fig)



# du_dt = Total_loss[-1]

# du_split = torch.split(du_dt,Npts)
# DUDT = torch.cat(tuple(k for k in du_split),1)
# DUDT = DUDT.cpu().detach().numpy()

idx_sample = np.random.choice(random_samples, size=10, replace=False)
idx_sample_phy = np.random.choice(random_samples_phy, size=10, replace=False)
for i in range(U.shape[1]):
    plt.figure()
    plt.plot(U_exa[:,i],label=f'Exact u{i}')
    plt.plot(U[:,i],label=f'Prédiction de u{i}')
   
    plt.plot(idx_sample,U_exa[idx_sample,i],label=f'Points observés u{i}',marker='o',linestyle='None')
    plt.plot(idx_sample_phy,[0 for j in range(len(idx_sample_phy))],label=f'Résidus u{i}',marker='x',linestyle='None')
    #plt.plot(DUDT[:,i],label = f"du/dt {i}")
    plt.xlabel('Temps')
    plt.ylabel('u')
    plt.legend()
    plt.savefig(PATH + f"/prediction_u{i}.png") #_{int(ratio*100)}
    plt.close()

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

indices = [0, 10, 24]  # U1, U6, U13 (numérotés à partir de 0)
labels  = [1, 6, 13]

for ax, i, lbl in zip(axes, indices, labels):
    ax.plot(U_exa[:, i],                                 label=f'Exact u{lbl}')
    ax.plot(U[:, i],                                     label=f'Prédiction de u{lbl}')
    ax.plot(idx_sample,     U_exa[idx_sample, i],        label=f'Points observés u{lbl}', marker='o', linestyle='None')
    ax.plot(idx_sample_phy, [0]*len(idx_sample_phy),     label=f'Résidus u{lbl}',               marker='x', linestyle='None')
    ax.set_ylabel(f'u{lbl}')
    ax.legend()

axes[-1].set_xlabel('Temps')
fig.tight_layout()
plt.savefig(PATH + "/prediction_u1_u6_u13.png")
plt.close(fig)


#torch.save(model.state_dict(),'/Odyssey/private/s26calme/code_stage/')


plt.figure()
plt.semilogy(Total_loss["loss"],label='Loss Total')
plt.semilogy(Total_loss["loss_physics_tracker"],label='Loss Physique totale')
#plt.plot(Total_loss[2],label='Colocation Loss')
plt.semilogy(Total_loss["loss_boundary_conditions_tracker"],label='Loss observations')
plt.plot(Total_loss["loss_initial_conditions_tracker"],label='initial Conditions Loss' )  
plt.xlabel('Iterations')
plt.ylabel('Losses')
plt.legend()
plt.savefig(PATH + f"losses.png") #_{int(ratio*100)}
plt.close()


plt.figure()
plt.semilogy(Total_loss["lmb_tracker_bc"],label='lmb obs')
#plt.semilogy(Total_loss["lmb_physics_trackeur"],label='lmb phy')
plt.semilogy(Total_loss["lmb_ic_trackeur"],label='lmb ic')
plt.xlabel('Iterations')
plt.ylabel('Lambda')
plt.legend()
plt.savefig(PATH + f"lambda.png") #_{int(ratio*100)}
plt.close()

n  = U_exa.shape[1]    # nb de composantes réelles (re + im par couche)
nb = U_exa.shape[0]        # nb de pas de temps

# U_exa et U sont de shape (temps, couches) → on transpose pour avoir (couches, temps)
Exa = U_exa.T   # shape (n/2, nb)
Pred = U.T      # shape (n/2, nb)

#K = range(1, n // 2 + 1)   # numéro de couche (pour la loi k^{-2/3})

# ── Variance log ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
ax.semilogy(range(n // 2), np.mean(Exa[0::2,:] ** 2 + Exa[1::2,:]**2,  axis=1), label='Vérité terrain')
ax.semilogy(range(n // 2), np.mean(Pred[0::2,:] ** 2 + Pred[1::2,:]**2, axis=1), label='Prédiction')
ax.semilogy(range(n // 2), [k ** (-2/3) for k in K], '--', alpha=0.5, label='$k^{-2/3}$')
ax.set_xlabel('Numéro de couche')
ax.set_ylabel(r'$\log(\langle|U_n|^2\rangle_T)$')
ax.legend()
fig.tight_layout()
plt.savefig(PATH + "/log_variance.png", format='png')
plt.close()

# ── Kurtosis log ──────────────────────────────────────────────────────────────
var_exa  = np.mean(Exa[0::2,:]**2 + Exa[1::2,:]** 2, axis=1)
var_pred = np.mean(Pred[0::2,:]**2 + Pred[1::2,:]**2, axis=1)

kurt_exa  = np.mean(Exa[0::2,:]** 4 + Exa[1::2,:]**4, axis=1) / var_exa  ** 2
kurt_pred = np.mean(Pred[0::2,:]**4 + Pred[1::2,:]**4, axis=1) / var_pred ** 2

fig, ax = plt.subplots()
ax.semilogy(range(n // 2), kurt_exa,  label='Vérité terrain')
ax.semilogy(range(n // 2), kurt_pred, label='Prédiction')
ax.set_xlabel('Numéro de couche')
ax.set_ylabel(r'$\log\!\left(\frac{\langle|U_n|^4\rangle_T}{\langle|U_n|^2\rangle_T^2}\right)$')
ax.legend()
fig.tight_layout()
plt.savefig(PATH + "/log_kurtosis.png", format='png')
plt.close()