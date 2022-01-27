import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import data2patches, patching

class DASDataset(Dataset):

    def __init__(self, id, dt, nt, mt, nd, md, dist_ini, dist_fin):
        data, self.__mean, self.__std = self.load_data(id, dt, dist_ini, dist_fin)
        self.X, self.__Lt, self.__Ld = data2patches(data, nt, mt, nd, md)
        self.dist = self.patches_dist(nt, mt, nd, md, dist_ini, dist_fin)

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.dist[idx]

    def load_data(self, id, dt, dist_ini, dist_fin):
        path = Path(__file__).parent.resolve()/('../../../data/Dt_'+str(dt))
        data = []
        with h5py.File(path/(str(id)+'.h5'),'r') as f:
            for i in range(dist_ini, dist_fin):
                data.append(f['dist'+str(i)][:])
        with h5py.File(path/('mean_std.h5'),'r') as f:
            mean = f['mean'][dist_ini:dist_fin]
            std = f['std'][dist_ini:dist_fin]
        data = np.array(data,dtype=np.int16).T
        mean, std = np.float32(mean), np.float32(std)
        return (data-mean)/std, mean, std

    
    def patches_dist(self, nt, mt, nd, md, dist_ini, dist_fin):
        Wt = (int)(np.ceil((self.__Lt-nt)/mt+1))
        dist = np.int16(patching(np.arange(dist_ini, dist_fin), nd, md))
        dist = np.float32((dist-dist_ini)/(dist_fin-dist_ini))
        return (np.tile(dist.T, Wt).T).reshape((-1,1,nd))

    @property
    def mean(self):
        return self.__mean

    @property
    def std(self):
        return self.__std

    @property
    def Lt(self):
        return self.__Lt

    @property
    def Ld(self):
        return self.__Ld