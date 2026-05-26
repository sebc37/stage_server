import torch
from torch.utils.data.sampler import Sampler


class SamplerOverGrid(Sampler):
    """
    Echantillonne m points consécutifs en temps pour chaque shell k
    entre k_min_grid et k_max_grid.
    """
    def __init__(self, m, k_min_grid, k_max_grid, Npts):
        self.m          = m
        self.k_min_grid = k_min_grid
        self.k_max_grid = k_max_grid
        self.Npts       = Npts

    def __iter__(self):
        for _ in range(self.__len__()):
            indice = torch.randint(0, self.Npts - self.m, (1,))
            nb_k   = self.k_max_grid - self.k_min_grid
            indices = torch.ones(nb_k * self.m, dtype=torch.int32)

            for i in range(self.m):
                indices[i] = indice + i + self.k_min_grid * self.Npts

            for k in range(1, self.k_max_grid - self.k_min_grid):
                for i in range(self.m):
                    indices[i + k * self.m] = int((k + self.k_min_grid) * self.Npts) + indice.item() + i

            yield indices.tolist()

    def __len__(self):
        return int(self.Npts // self.m)
