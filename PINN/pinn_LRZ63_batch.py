import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import  architecture
import tqdm
from tqdm import contrib

torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = "/Odyssey/private/s26calme/code_stage/PINN/"

list_of_datasets = glob.glob(PATH+"dataset_*.npy")

# faire différent dataset : parti facile de l'attracteur et point d'equilibre 
# regarder avec des (ti,xi,yi,zi) qui predise des(xj,yj,zj) avec j>i
# faire une fonction closure pour LBFGS



#############################
# modèle lorenz 63 avec RK4 #
#############################
nb    = 1000 # number of times
n=3
time  = np.array(range(nb))

sigma=10
rho=28
beta=8/3


### define the nonlinear dynamic system (Lorenz-63) using the Runge-Kutta integration method
def m(x_past,dT,sigma,rho,beta):
    
    # physical parameters
    # dT=0.01
    # sigma=10
    # rho=28
    # beta=8/3
    
    # Runge-Kutta (4,5) integration method
    X1 = np.copy(x_past)
    k1 = np.zeros(X1.shape)
    k1[0] = sigma*(X1[1] - X1[0])
    k1[1] = X1[0]*(rho-X1[2]) - X1[1]
    k1[2] = X1[0]*X1[1] - beta*X1[2]
    X2 = np.copy(x_past+k1/2*dT)
    k2 = np.zeros(x_past.shape)
    k2[0] = sigma*(X2[1] - X2[0])
    k2[1] = X2[0]*(rho-X2[2]) - X2[1]
    k2[2] = X2[0]*X2[1] - beta*X2[2]   
    X3 = np.copy(x_past+k2/2*dT)
    k3 = np.zeros(x_past.shape)
    k3[0] = sigma*(X3[1] - X3[0])
    k3[1] = X3[0]*(rho-X3[2]) - X3[1]
    k3[2] = X3[0]*X3[1] - beta*X3[2]
    X4 = np.copy(x_past+k3*dT)
    k4 = np.zeros(x_past.shape)
    k4[0] = sigma*(X4[1] - X4[0])
    k4[1] = X4[0]*(rho-X4[2]) - X4[1]
    k4[2] = X4[0]*X4[1] - beta*X4[2]

    # return the state in the near future
    x_future = x_past + dT/6.*(k1+2*k2+2*k3+k4)
    
    return x_future


def make_dataset(nb,n,CI,dT,sigma,rho,beta,name):
    
    if PATH+f"dataset_{name}_LR63.npy" not in list_of_datasets:
        x = np.zeros((n,nb))
        x[:,0] = CI
        for t in range(1,nb):
            x[:,t] = m(x[:,t-1],dT,sigma,rho,beta)
        
        
        np.save(PATH+f"dataset_{name}_LR63.npy",x)
        return 0

make_dataset(nb=27000,n=n,CI=np.array([8,0,30]),dT=1e-3,sigma=10,rho=28,beta=8/3,name="long_attractor")
for i in range(3):
    make_dataset(nb=3000,n=n,CI=np.array([0,0,0])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"zero_attractor_{i}")
    make_dataset(nb=3000,n=n,CI=np.array([-np.sqrt(beta*(rho-1)),-np.sqrt(beta*(rho-1)),rho-1])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"equil_{i}")
    make_dataset(nb=3000,n=n,CI=np.array([np.sqrt(beta*(rho-1)),np.sqrt(beta*(rho-1)),rho-1])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"equilibrium_{i}")
for i in range(9):
    make_dataset(nb=9000,n=n,CI=np.array([8,0,30])+np.random.rand(3)*0.01,dT=1.e-3,sigma=10,rho=28,beta=8/3,name=f"short_attractor_{i}")

# x = np.zeros((n,nb))
# x[:,0] = np.array([8,0,30])

# for t in range(1,nb):
#     x[:,t] = m(x[:,t-1])
easy_dataset = [p for p in list_of_datasets if(('zero' in p) or ('equil' in p)) ]
medium_dataset = [p for p in list_of_datasets if('short' in p)]
hard_dataset = [p for p in list_of_datasets if('long' in p)]

def plot_attractor(p):
    x = np.load(PATH+p)
    xyzs = x.T
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*xyzs.T, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor {p.split('.')[0]}")

def plot_attractor_bis(x,y,name):
    
    ax = plt.figure().add_subplot(projection='3d')

    ax.plot(*x, lw=0.5,label='True')
    ax.plot(*y, lw=0.5,label='prediction')
    #ax.plot(*x[0],'ro', label="IC")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor {name}")
    plt.legend()
    plt.savefig(PATH + f'Lorenz_Attractor_{name}')


def plot_attractor_(x,y,name):
    xyzs = x.T
    ax = plt.figure().add_subplot(projection='3d')
    # Unpack columns — works for any (T/dt, 3) numpy array
    ax.plot(xyzs[:, 0], xyzs[:, 1], xyzs[:, 2],
        lw=0.5, alpha=0.85, color="royalblue", label="Truth")

    ax.plot(y[:, 0], y[:, 1], y[:, 2],
        lw=0.5, alpha=0.85, color="tomato",    label="Prediction")

# Mark starting points
    # ax.scatter(*xyzs[0], color="royalblue", s=40, zorder=5, label="IC 1")
    # ax.scatter(*y[0], color="tomato",    s=40, zorder=5, label="IC 2")
    # ax.plot(*xyzs.T, lw=0.5,label='real')
    # ax.plot(*y,lw=0.5,label='prediction')
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    ax.set_title(f"Lorenz Attractor {name}")
    ax.legend()
    plt.savefig(PATH + f'Lorenz_Attractor_{name}')
    #plt.show()

list_of_datasets = glob.glob(PATH+"dataset_*.npy")

list_dt = [1e-3 for i in range(len(list_of_datasets))]
# for p in list_of_datasets:
#     plot_attractor(p)
#################################################
# Dataset LR63 avec BC, IC, Collocation points  #
#################################################


# x = np.load(easy_dataset[0])

# x_bc = x[0,:]
# x_ic = x[:,0]



# iy_cl = np.random.choice(range(nb), 500)
# iz_cl = np.random.choice(range(nb), 500)

# # y_cl = np.c_[iy_cl,np.array([1 for _ in range(500)],dtype=np.float32),x[1,iy_cl]]
# # z_cl = np.c_[iz_cl,np.array([2 for _ in range(500)],dtype=np.float32),x[2,iz_cl]]
# i_cl =np.c_[iy_cl,iz_cl]
# x_cl = np.c_[x[1,iy_cl],x[2,iz_cl]]


# ### dataset mis sous forme de tensor
# X_BC = torch.tensor(x_bc,dtype=torch.float32) # x
# X_CL = torch.tensor(x_cl,dtype=torch.float32) # y,z point aléatoire
# X_IC = torch.tensor(x_ic,dtype=torch.float32) # x(0),y(0),z(0)
# X_ALL = torch.tensor(x,dtype=torch.float32) # x,y,z 
# ### grille 
# T = torch.tensor(time,dtype=torch.float32)

# X = torch.stack((T,torch.zeros_like(T,dtype=torch.float32),torch.tensor(x[0,:],dtype=torch.float32)))
# Y = torch.stack((T,torch.ones_like(T,dtype=torch.float32),torch.tensor(x[1,:],dtype=torch.float32)))
# Z = torch.stack((T,torch.ones_like(T,dtype=torch.float32)*2,torch.tensor(x[2,:],dtype=torch.float32)))

# FULL_GRID = torch.cat((X,Y,Z),dim=1).mT


#T_CL = torch.tensor(i_cl,dtype=torch.float32)#torch.stack((torch.tensor(i_cl,dtype=torch.float32),torch.tensor(i_cl,dtype=torch.float32)),dim=0)
# T_IC = torch.stack((torch.zeros(n,dtype=torch.float32),torch.tensor([i for i in range(n)],dtype=torch.float32)))
# POS = torch.zeros_like(T)



#dataset_bc = torch.utils.data.TensorDataset(torch.stack((T,POS,X_BC)).mT)
#dataset_cl = torch.utils.data.TensorDataset(X_CL) # indice dans T_CL ou X_CL donne y=0 ou z=1 


# plt.figure()
# plt.plot(time,x_bc.T)
# plt.plot(i_cl,x_cl.T,'*')
# plt.show()

# plt.figure()
# plt.plot(T,X_BC)
# plt.plot(T_CL.mT,X_CL.mT,'*')
# plt.show()

##############
# Dataloader #
##############

# trainloader_bc = torch.utils.data.DataLoader(dataset_bc, batch_size=128, shuffle=True, drop_last=False)
# trainloader_cl = torch.utils.data.DataLoader(dataset_cl, batch_size=128, shuffle=True, drop_last=False)

# for idx,d in enumerate(trainloader_bc):
#     print(type(d))
#     print(d)


###############################
# loss BC, IC ,CL et physique #
###############################

sigma=10
rho=28
beta=8/3

def loss_bc(x,y):
    return torch.mean((x-y)**2)

def loss_ic(x,y):
    return torch.mean((x-y)**2)


alpha = 0.9 #moving average normalization loss
def loss_phy(x,y,z,T,lmb_x,lmb_y,lmb_z):
    
    #dxyz = torch.autograd.grad(XYZ,T,torch.ones_like(XYZ),create_graph=True)
    # dx = dxyz[:,0:1]
    # dy = dxyz[:,1:2]
    # dz = dxyz[:,2:3]
    dx = grad(x,T).unsqueeze(-1)
    dy = grad(y,T).unsqueeze(-1)
    dz = grad(z,T).unsqueeze(-1)

    res_x = dx - sigma*(y-x)
    res_y = dy - (x*(rho-z) - y)
    res_z = dz - (x*y - beta*z)
    loss_phy = lmb_x*torch.mean(res_x**2) + lmb_y*torch.mean(res_y**2) + lmb_z*torch.mean(res_z**2)
    #dydt = torch.autograd.grad(x[:,1],grid,torch.ones_like(x),create_graph=True)[0][:,0]
    # LOSS_PHY = torch.zeros_like(x,dtype=torch.float32).to(device)
    # LOSS_PHY[0:nb] = dx[0:nb] - sigma*(x[nb:2*nb] - x[0:nb])
    # LOSS_PHY[nb:2*nb] = dx[nb:2*nb] - (rho*x[0:nb] - x[nb:2*nb] - x[0:nb]*x[2*nb:3*nb])
    # LOSS_PHY[2*nb:3*nb] = dx[2*nb:3*nb] - (x[0:nb]*x[nb:2*nb]- beta*x[2*nb:3*nb])
    # loss = torch.mean(LOSS_PHY**2)
    return loss_phy,res_x,res_y,res_z

# a = loss_bc(X_BC,X_BC)
# print("a")

def grad(y, t):
    return torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),   # sum over N points
        create_graph=True,                 # ← keep graph for higher-order or loss backprop
        retain_graph=True
    )[0]   



#############
#   Train   #
#############

class Train_PINN():

    def __init__(self,learning_rate,nbr_iteration,w_1,w_2,w_3,optim,datasets,dts):
        self.learning_rate = learning_rate
        self.nbr_iteration = nbr_iteration
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.datasets=datasets
        self.dts = dts
        
        # if type(optim) != None:
        #     self.optimizer = optim
        # else:
        self.optimizer = torch.optim.Adam(model.parameters(),lr = self.learning_rate)
        
    def train(self):
        
        loss = np.zeros((self.nbr_iteration,1))
        loss_bc_tracker = np.zeros((self.nbr_iteration,1))
        loss_ic_tracker = np.zeros((self.nbr_iteration,1))
        loss_cl_tracker = np.zeros((self.nbr_iteration,1))
        loss_phy_tracker = np.zeros((self.nbr_iteration,1))
        lamnda_record = np.zeros((self.nbr_iteration,5))

        # list_dataset = []
        # list_dataloader = []
        
        # for d,t in zip(dataset,dt):
        #     list_dataset.append(architecture.Lorenz_Dataset(d,t))
        
        # for dat in list_dataset:
        #     list_dataloader.append(architecture.DataLoader(dat,batch_size=dat.__len__()/10,shuffle=True))
        liste_dataset= [architecture.Lorenz_Dataset(self.datasets[0][i],self.dts[i],1) for i in range(len(self.datasets[0]))]
        #dataset_easy = torch.utils.data.StackDataset(l)
        #temps,xyz = l[0][2997]
        print(liste_dataset)
        liste_dataloader = [architecture.DataLoader(elm,batch_size=10,shuffle=True) for elm in liste_dataset]

        lmb_x = 1
        lmb_y = 1
        lmb_z = 1
        lmb_ic = 1
        lmb_bc = 1
        #dataloader_test = architecture.DataLoader(l[0],batch_size=10,shuffle=True)
        
       

        # for i in range(10):
        #     temps,xyz = next(iter(dataloader_test))
            
        # data =[]
        #data = architecture.DataLoader(dataset_easy,batch_sampler=torch.utils.data.SequentialSampler(dataset_easy.datasets[0]))
        #data = architecture.DataLoader(architecture.Multilple_Lorenz(self.datasets,self.dts,900))
        
        for iteration in tqdm.tqdm(range(self.nbr_iteration)):
            self.optimizer.zero_grad()
            #index = int(np.random.random*len(list_dataloader))
            
            loss_batch=0
            loss_batch_phy = 0
            loss_ic = 0
            c = 0
            for dataloader in liste_dataloader:
                
                for idx,d in enumerate(dataloader):
                    # conditions initiales
                    ic = liste_dataset[c]
                    print(ic)
                    print(ic[0])
                    data_ic = ic[0][1].to(device)#[0,:].to(device)
                    u_ic_pred = model(torch.tensor([0,0,0],dtype=torch.float32).unsqueeze(-1).to(device))
                    
                    # batches
                    times = d[0].requires_grad_(True).to(device)
                    xyzs = d[1].to(device)
                    u_pred = model(times.unsqueeze(-1))

                    loss_ic = torch.mean((data_ic-u_ic_pred)**2)
                    loss_batch = torch.mean((u_pred-xyzs)**2)
                    loss_batch_phy,res_x,res_y,res_z = loss_phy(x=u_pred[:,0:1],y=u_pred[:,1:2],z=u_pred[:,2:3],T=times,lmb_x=lmb_x,lmb_y=lmb_y,lmb_z=lmb_z)
                    loss_batch_phy = loss_batch_phy/n
                    total_loss_batch = lmb_bc*loss_batch + loss_batch_phy + 6000*lmb_ic*loss_ic
                    
                    # grad_res_x = torch.autograd.grad(res_x, model.parameters(),grad_outputs=torch.ones_like(res_x), retain_graph=True)
                    # grad_res_y = torch.autograd.grad(res_y, model.parameters(),grad_outputs=torch.ones_like(res_y), retain_graph=True)
                    # grad_res_z = torch.autograd.grad(res_z, model.parameters(),grad_outputs=torch.ones_like(res_z), retain_graph=True)
                    # grad_Loss_batch = torch.autograd.grad(u_pred, model.parameters(),grad_outputs=torch.ones_like(u_pred), retain_graph=True)
                    # grad_ic = torch.autograd.grad(u_ic_pred, model.parameters(),grad_outputs=torch.ones_like(u_ic_pred), retain_graph=True)

                    # norm_grad_x = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_res_x])).detach()
                    # norm_grad_y = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_res_y])).detach()
                    # norm_grad_z = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_res_z])).detach()
                    # norm_grad_L_bc = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_Loss_batch])).detach()
                    # norm_grad_ic = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_ic])).detach()
                    
                    # total_lmb = norm_grad_L_bc + norm_grad_x + norm_grad_y + norm_grad_z + norm_grad_ic
                    # lmb_x_new = total_lmb/ (norm_grad_x+ 1e-10)
                    # lmb_y_new = total_lmb/ (norm_grad_y + 1e-10)
                    # lmb_z_new = total_lmb/ (norm_grad_z + 1e-10)
                    # lmb_bc_new = total_lmb/ (norm_grad_L_bc + 1e-10)
                    # lmb_ic_new = total_lmb/ (norm_grad_ic + 1e-10)

                    # lmb_ic = alpha*lmb_ic + (1-alpha)*lmb_ic_new
                    # lmb_x = alpha*lmb_x + (1-alpha)*lmb_x_new
                    # lmb_y = alpha*lmb_y + (1-alpha)*lmb_y_new
                    # lmb_z = alpha*lmb_z + (1-alpha)*lmb_z_new
                    # lmb_bc = alpha*lmb_bc + (1-alpha)*lmb_bc_new

                    total_loss_batch.backward()
                    self.optimizer.step()



                    # for t,xyz in zip(times,xyzs):
                    #     t = t.unsqueeze(-1).requires_grad_().to(device)
                    #     u_ic_pred = model(torch.tensor([0,0,0],dtype=torch.float32).unsqueeze(-1).to(device))
                    #     u_exact = xyz.to(device)
                    #     u_pred = model(t)
                    #     loss_ic = loss_ic + torch.mean((u_ic_pred-data_ic)**2)/n
                    #     loss_batch = loss_batch + (torch.mean((u_exact-u_pred)**2))/10 # mettre 10 en paramètre
                    #     loss_batch_phy = loss_batch_phy + (loss_phy(x=u_pred[:,0:1],y=u_pred[:,1:2],z=u_pred[:,2:3],T=t))/n
                    #     total_loss_batch = (loss_batch_phy + loss_batch + loss_ic)
                    #     total_loss_batch.backward()
                    #     self.optimizer.step()

                # c+=1  
                # total_loss_batch = (loss_batch_phy/3000 + loss_batch/3000 + loss_ic/3000)/len(list_dataloader)
                # total_loss_batch.backward()
                # self.optimizer.step()

            # Save losses
            lamnda_record[iteration,:] = np.array([lmb_bc#.cpu().detach().numpy()
                                                  ,lmb_ic#.cpu().detach().numpy()
                                                  ,lmb_x#.cpu().detach().numpy()
                                                  ,lmb_y#.cpu().detach().numpy()
                                                  ,lmb_z#.cpu().detach().numpy()
                                                  ])
            loss[iteration]=total_loss_batch.cpu().detach().numpy()
            loss_bc_tracker[iteration] = loss_batch.cpu().detach().numpy()
            #loss_cl_tracker[iteration] = loss_collocation_conditions.cpu().detach().numpy()
            loss_phy_tracker[iteration] = loss_batch_phy.cpu().detach().numpy()
            loss_ic_tracker[iteration] = loss_ic.cpu().detach().numpy()
        return loss,loss_ic_tracker,loss_bc_tracker,loss_cl_tracker,loss_phy_tracker,lamnda_record

            
        #     # for idx,dataload in enumerate(data):
        #     #     print(idx)
        #     #     print(dataload)
        #     #     for id,sample in enumerate(dataload[0]):
        #     #         print(id)
        #     #         print(sample)

        #     # #initial conditions Loss
        #     # initial_train_data = torch.cat((T_IC,X_IC.unsqueeze(0))).mT.to(device)#torch.tensor(initial_train_dataset)
        #     # u_pd_ini = model(initial_train_data[:, 0:2])
        #     # u_exa_ini = initial_train_data[:,2:3]
        #     # loss_initital_conditions = self.w_1*torch.mean((u_pd_ini-u_exa_ini)**2)
           
        #     #Boundary conditions Loss
        #     # boundary_train_data = dataset_bc
        #     # u_pd_bou = model(boundary_train_data[:,0:2][0].to(device))
        #     # u_exa_bou = boundary_train_data[:,2:3][0].to(device)
        #     # loss_boundary_conditions = self.w_2*torch.mean((u_pd_bou-u_exa_bou)**2)
            
        #     t = T.unsqueeze(-1).requires_grad_().to(device)
        #     u_pd_all = model(t)
        #     u_exa_all = X_ALL.mT.to(device)

        #     loss_boundary_conditions = self.w_2*torch.mean((u_pd_all-u_exa_all)**2)


        #     # Collocation conditions Loss
        #     # collocation_train_data = dataset_cl
        #     # u_pd_cl = model(collocation_train_data[:,0:2][0].to(device))
        #     # u_exa_cl = collocation_train_data[:,2:3][0].to(device)
        #     # loss_collocation_conditions = self.w_3*torch.mean((u_pd_cl-u_exa_cl)**2)

        #     #Physical Loss
        #     # train_data = FULL_GRID[:,0:2].requires_grad_(True).to(device)
        #     # u_pd = model(train_data)
        #     #a = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)
        #     #u_t = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]

        #     # u_x = torch.autograd.grad(u_pd, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]
        #     # u_xx = torch.autograd.grad(u_x, train_data, torch.ones_like(u_pd), create_graph=True)[0][:,0:1]
        #     #physics = u_t + u_pd*u_x - nu*u_xx
        #     u_pd_all.requires_grad_()
        #     loss_physics = self.w_3*loss_phy(x=u_pd_all[:,0:1],y=u_pd_all[:,1:2],z=u_pd_all[:,2:3],T=t)

        #     #Total Loss
        #     total_loss =  loss_boundary_conditions + loss_physics#+ loss_physics #loss_initital_conditions +  loss_collocation_conditions
        #     total_loss.backward()
        #     self.optimizer.step()
            
        #     # Save losses
        #     loss[iteration]=total_loss.cpu().detach().numpy()
        #     loss_bc_tracker[iteration] = loss_boundary_conditions.cpu().detach().numpy()
        #     #loss_cl_tracker[iteration] = loss_collocation_conditions.cpu().detach().numpy()
        #     loss_phy_tracker[iteration] = loss_physics.cpu().detach().numpy()
        #     #loss_ic_tracker[iteration] = loss_initital_conditions.cpu().detach().numpy()
        # return loss,loss_ic_tracker,loss_bc_tracker,loss_cl_tracker,loss_phy_tracker,self.optimizer.state_dict()


#u_t = torch.autograd.grad(u_pd, grid_train_data, torch.ones_like(u_pd), create_graph=True)[0][:,1:2].to(device)
easy_dataset = [p for p in list_of_datasets if(('zero' in p))  ] #or ('equil' in p))
medium_dataset = [p for p in list_of_datasets if('short' in p)]
hard_dataset = [p for p in list_of_datasets if('long' in p)]
datasets_list = [easy_dataset,medium_dataset,hard_dataset]

#############
#   modèle  #
# ###########            

model = architecture.GOY_PINN(1,3,500,3,batch_size=(32,300,10),ic_size=(32,3,1))
model.to(device)

lr = 5*1.0e-5
nb_iter = 100
t =Train_PINN(learning_rate=lr,nbr_iteration=nb_iter,w_1=1,w_2=1,w_3=1,optim=None,datasets=datasets_list,dts=list_dt)
l,li,lb,lc,lp,lambda_enr =  t.train()

T = torch.arange(0,3000*1e-3,1e-3,dtype=torch.float32)
U_pred = model(T.unsqueeze(-1).to(device))
print(U_pred.shape)
U = U_pred.cpu().detach().numpy()#U_pred.view((n,nb)).mT.cpu().detach().numpy()
print(np.shape(U))
real_data = np.load(easy_dataset[0])
#################
# loop training #
#################

# séparer les dataset 


# for dataset in datasets_list:
    
#     D = []
#     for l in dataset:
#         D.append(torch.tensor(np.load(PATH + l),dtype=torch.float64,requires_grad=True))

    
#     t = Train_PINN(..., dataset=D)
    
    # training on each type of dataset
    #t = Train_PINN(nbr_iteration=nb_iter,optim=)


    # torch.save({
    #         'epoch': nb_iter,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': dict_optim,
    #         'loss': l,
            
    #         }, PATH + "model_save")


################
# saving model #
################

# torch.save({
#             'epoch': nb_iter,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': dict_optim,
#             'loss': l,
            
#             }, PATH + "model_save")



#############
#   plots   #
#############

# prediction vs realité
#plot_attractor_(U,real_data,'zero')
plot_attractor_bis(real_data,U.T,'pred_zero')

plt.figure()

plt.plot(T,real_data[0,:],label='x real')
plt.plot(T,real_data[1,:],label='y real')
plt.plot(T,real_data[2,:],label='z real')

plt.plot(T,U.T[0,:],'--',label='x pred',alpha=0.7)
plt.plot(T,U.T[1,:],'--',label='y pred',alpha=0.7)
plt.plot(T,U.T[2,:],'--',label='z pred',alpha=0.7)
plt.legend()
plt.savefig(PATH + 'x_y_z')
#plot_attractor_bis(real_data,'real_zero')
# plt.figure()
# plt.plot(time,x_bc.T)
# plt.plot(i_cl.T,x_cl.T,'*')
# plt.plot(time,U[:,0],label="x pred")
# plt.plot(time,U[:,1],label="y pred")
# plt.plot(time,U[:,2],label="z pred")
# plt.legend()
# plt.savefig(PATH+"prediction")
#plt.show()
nom_lmb = ['bc','ic','x','y','z']
plt.figure()
for i in range(lambda_enr.shape[1]):
    plt.yscale('log')
    plt.plot(lambda_enr[:,i],label=f'lamda_{nom_lmb[i]}')
    plt.legend()
plt.savefig(PATH + 'lambda_norm')
# losses
plt.figure()
plt.yscale('log')
plt.plot(l,label="Total loss")
plt.plot(li,label="IC loss")
plt.plot(lb,label="BC loss")
plt.plot(lc,label="CL loss")
plt.plot(lp,label="PHY loss")
plt.legend()
plt.savefig(PATH+"losses")
#plt.show()

