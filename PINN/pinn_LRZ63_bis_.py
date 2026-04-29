import numpy as np
import matplotlib.pyplot as plt
import torch
import  architecture
import tqdm

torch.manual_seed(119)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = r"/Odyssey/private/s26calme/code_stage/PINN/LRZ63/"


#############################
# modèle lorenz 63 avec RK4 #
#############################
nb    = 1000 # number of times
n     = 3
time  = np.array(range(nb))*0.01

### define the nonlinear dynamic system (Lorenz-63) using the Runge-Kutta integration method
def m(x_past):
    
    # physical parameters
    dT=0.01
    sigma=10
    rho=28
    beta=8/3
    
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


x = np.zeros((n,nb))
x[:,0] = np.array([8,0,30])

for t in range(1,nb):
    x[:,t] = m(x[:,t-1])

###############################
# plots Lorenz trajectory     #

################################
ax = plt.figure().add_subplot(projection='3d')

ax.plot(*x, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

#################################################
# Dataset LR63 avec BC, IC, Collocation points  #
#################################################

x_bc = x[0,:] # x(t) for all t
x_ic = x[:,0] # x(0),y(0),z(0)



iy_cl = np.random.choice(range(nb), 500) # t des y(t) points aléatoires
iz_cl = np.random.choice(range(nb), 500) # t des z(t) points aléatoires


i_cl =np.c_[iy_cl,iz_cl] 
x_cl = np.c_[x[1,iy_cl],x[2,iz_cl]] # y(t),z(t) point aléatoire

### dataset mis sous forme de tensor
X_BC = torch.tensor(x_bc,dtype=torch.float32) # x
X_CL = torch.tensor(x_cl,dtype=torch.float32) # y,z point aléatoire
X_IC = torch.tensor(x_ic,dtype=torch.float32) # x(0),y(0),z(0)
X_ALL = torch.tensor(x,dtype=torch.float32) # x,y,z 

### grille 
T = torch.tensor(time,dtype=torch.float32)


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

def loss_phy(x,y,z,T,wx,wy,wz):
    
   
    dx = grad(x,T)
    dy = grad(y,T)
    dz = grad(z,T)

    res_x = dx - sigma*(y-x)
    res_y = dy - (x*(rho-z) - y)
    res_z = dz - (x*y - beta*z)
    loss_phy = wx*torch.mean(res_x**2) + wy*torch.mean(res_y**2) + wz*torch.mean(res_z**2)
   
    return loss_phy,res_x,res_y,res_z



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

    def __init__(self,learning_rate,nbr_iteration,w_1,w_2,w_3):
        self.learning_rate = learning_rate
        self.nbr_iteration = nbr_iteration
        self.w_1 = w_1
        self.w_2 = w_2
        self.w_3 = w_3
        self.wx = 1
        self.wy = 1
        self.wz = 1
        self.optimizer = torch.optim.Adam(model.parameters(),lr = self.learning_rate)

    def train(self):

        model.train()
        loss = np.zeros((self.nbr_iteration,1))
        lbc = np.zeros((int(self.nbr_iteration/1000),1))
        lx= np.zeros((int(self.nbr_iteration/1000),1))
        ly = np.zeros((int(self.nbr_iteration/1000),1))
        lz = np.zeros((int(self.nbr_iteration/1000),1))
        loss_bc_tracker = np.zeros((self.nbr_iteration,1))
        loss_phy_tracker = np.zeros((self.nbr_iteration,1))
        t = T.unsqueeze(-1).requires_grad_().to(device)

        for iteration in tqdm.tqdm(range(self.nbr_iteration)):
            self.optimizer.zero_grad()

      
            u_pd_all = model(t)
            u_exa_all = X_ALL.mT.to(device)

            loss_boundary_conditions = self.w_2*torch.mean((u_pd_all-u_exa_all)**2)


    
            u_pd_all.requires_grad_()
            loss_physics,res_x,res_y,res_z = self.w_3*loss_phy(x=u_pd_all[:,0:1],y=u_pd_all[:,1:2],z=u_pd_all[:,2:3],T=t,wx=self.wx,wy=self.wy,wz=self.wz)

            if iteration % 1000 == 0:
               
               grad_res_x = torch.autograd.grad(res_x, model.parameters(),grad_outputs=torch.ones_like(res_x), retain_graph=True)
               grad_res_y = torch.autograd.grad(res_y, model.parameters(),grad_outputs=torch.ones_like(res_y), retain_graph=True)
               grad_res_z = torch.autograd.grad(res_z, model.parameters(),grad_outputs=torch.ones_like(res_z), retain_graph=True)

               grad_Loss_bc = torch.autograd.grad(u_pd_all, model.parameters(),grad_outputs=torch.ones_like(u_pd_all), retain_graph=True)
               
            #   print("grad_Loss_phy : ",grad_Loss_phy.__len__(),grad_Loss_phy[0].shape,grad_Loss_phy[1].shape,grad_Loss_phy[2].shape,grad_Loss_phy[3].shape)
            #   print("grad_Loss_bc : ",grad_Loss_bc.__len__(),grad_Loss_bc[0].shape,grad_Loss_bc[1].shape,grad_Loss_bc[2].shape,grad_Loss_bc[3].shape)
               a = torch.stack((torch.cat([g.view(-1) for g in grad_res_x]),torch.cat([g.view(-1) for g in grad_res_y]),torch.cat([g.view(-1) for g in grad_res_z])),dim=1)
               b = torch.matmul(a,a.mT)

               print("grad_res_x : ",a.shape)
               print("NTK : ",b.shape )

               norm_grad_x = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_res_x]))
               norm_grad_y = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_res_y]))
               norm_grad_z = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_res_z]))
               norm_grad_L_bc = torch.linalg.norm(torch.cat([g.view(-1) for g in grad_Loss_bc]))
               total_lmb = norm_grad_L_bc + norm_grad_x + norm_grad_y + norm_grad_z
               lmb_x = total_lmb/ (norm_grad_x+ 1e-10)
               lmb_y = total_lmb/ (norm_grad_y + 1e-10)
               lmb_z = total_lmb/ (norm_grad_z + 1e-10)
               lmb_bc = total_lmb/ (norm_grad_L_bc + 1e-10)
               self.w_2 = lmb_bc
               self.wx = lmb_x
               self.wy = lmb_y
               self.wz = lmb_z
               
               lbc[int(iteration/1000)] = lmb_bc.cpu().detach().numpy()
               lx[int(iteration/1000)] = lmb_x.cpu().detach().numpy()
               ly[int(iteration/1000)] = lmb_y.cpu().detach().numpy()
               lz[int(iteration/1000)] = lmb_z.cpu().detach().numpy()
               #print(f"Iteration {iteration} - Lambda BC: {lmb_bc:.4f}, Lambda PHY: {lmb_phy:.4f}")


            #Total Loss
            total_loss =  loss_boundary_conditions + loss_physics
            total_loss.backward()
            self.optimizer.step()
            
            # Save losses
            loss[iteration]=total_loss.cpu().detach().numpy()
            loss_bc_tracker[iteration] = loss_boundary_conditions.cpu().detach().numpy()
            loss_phy_tracker[iteration] = loss_physics.cpu().detach().numpy()
        
        return loss,loss_bc_tracker,loss_phy_tracker,lbc,lx,ly,lz


#u_t = torch.autograd.grad(u_pd, grid_train_data, torch.ones_like(u_pd), create_graph=True)[0][:,1:2].to(device)
#############
#   modèle  #
# ###########            

model = architecture.GOY_PINN(1,3,64,4)
model.to(device)

lr = 1.0e-4
nb_iter = 3000
t =Train_PINN(learning_rate=lr,nbr_iteration=nb_iter,w_1=1,w_2=1,w_3=1)
l,li,lb,lc,lx,ly,lz =  t.train()

U_pred = model(T.unsqueeze(-1).to(device))
U = U_pred.cpu().detach().numpy()


#############
#   plots   #
#############

# prediction vs realité
plt.figure()
plt.plot(time,x_bc.T)
#plt.plot(i_cl.T,x_cl.T,'*')
plt.plot(time,U[:,0],label="x pred")
plt.plot(time,U[:,1],label="y pred")
plt.plot(time,U[:,2],label="z pred")
plt.legend()
plt.savefig(PATH+"prediction_norm")
#plt.show()

# losses
plt.figure()
plt.plot(l,label="Total loss")
plt.plot(li,label="loss data")
plt.plot(lb,label="loss phy")
# plt.plot(lc,label="CL loss")
# plt.plot(lp,label="PHY loss")
plt.legend()
plt.savefig(PATH+"losses_bis_phy")
#plt.show()

plt.figure()
plt.plot(lc,label="lambda boundary")
plt.plot(lx,label="lambda physic x")
plt.plot(ly,label="lambda physic y")
plt.plot(lz,label="lambda physic z")
plt.legend()
plt.savefig(PATH+"lmb")
