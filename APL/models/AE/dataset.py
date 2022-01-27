import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from ..utils import data2patches

class DASDataset(Dataset):

    def __init__(self, id, dt, nt, mt, nd, md, dist_ini, dist_fin):
        data, self.__mean, self.__std = self.load_data(id, dt, dist_ini, dist_fin)
        self.X, self.__Lt, self.__Ld = data2patches(data, nt, mt, nd, md)

    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx]

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