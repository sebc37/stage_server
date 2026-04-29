from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
import torch

class SamplerOverGrid(Sampler):
    '''
    m le nombre de points pris à partir de l'indice pris au hasard entre 0 et Npts
    k le nombre de shells où l'on souhaite évaluer des ppoints 
    Npts le nombre de pas de temps 
    
    retourne m points consécutifs dans le temps à partir d'un rang au hasard dans le temps
    pour chaque shell

    '''
    def __init__(self, m,k_min_grid,k_max_grid,Npts):
        self.m = m
        self.k_min_grid = k_min_grid
        self.k_max_grid = k_max_grid
        self.Npts = Npts
    def __iter__(self):
        for _ in range(self.__len__()):
            indice = torch.randint(0,self.Npts-self.m,(1,))
            nb_k = self.k_max_grid - self.k_min_grid
            indices = torch.ones(nb_k*self.m,dtype=torch.int32)
            #if self.k_min_grid == 0:
            for i in range(0,self.m):
                indices[i] = indice + i + self.k_min_grid*self.Npts
            for k in range(1,self.k_max_grid-self.k_min_grid):
                for i in range(0,self.m):        
                    indices[i+ k*self.m] = int((k+self.k_min_grid)*(self.Npts)) + indice.item() + i
        # else : 
           
        #     for k in range(self.k_min_grid,self.k_max_grid):
        #         for i in range(0,self.m):        
        #             indices[i+ (k-self.k_min_grid)*self.m] = int(k*(self.Npts)) + indice.item() + i
            print(f'sampler {indices}')
            yield indices.tolist()
                
    def __len__(self):
        return int(self.Npts // self.m)

# class MyDataset(Dataset):
#     def __init__(self, tensor):
#         self.tensor = tensor  # shape (k, n, 3)
#         self.k, self.n = tensor.shape

#     def __getitem__(self, indices):
#         return self.tensor[indices, :]

#     def __len__(self):
#         return 1


# tensor = torch.arange(0, 500)
# tensor_bis = torch.ones(500,2)
# tensor_bis[:,1] = tensor
# sampler = SamplerOverGrid(Npts=100,k=5, m=10)
# batch_sampler = BatchSampler(sampler, batch_size=10, drop_last=False)
# dataset = MyDataset(tensor_bis)
# loader = DataLoader(dataset, batch_sampler=batch_sampler)

# for batch in loader:
#     print(batch)
