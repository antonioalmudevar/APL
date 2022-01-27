import numpy as np
from pathlib import Path
import h5py
from .read_files import select_datafile, read_info

DEFAULT_N_SILENCES = 3

def read_raw_silence(args):
    data = []
    Lt = 0
    for i in (range(DEFAULT_N_SILENCES)):
        id_file = 'silencio'+str(i+1)
        filename = select_datafile(id_file, args.data_path)
        Lt_tmp, Ld = read_info(filename)
        Lt+=Lt_tmp
        with open(filename,'rb') as f:
            data_tmp = np.frombuffer(f.read(), dtype=np.int16).reshape(Lt_tmp,Ld)
        data.append(data_tmp)
    data = np.array(data).reshape((Lt,Ld))
    data = np.reshape(data,(Lt//args.dt,args.dt,Ld))
    data = np.swapaxes(data,0,1)
    #np.random.shuffle(data)
    data = np.reshape(data,(Lt,Ld))
    mean, std = np.mean(data[:10*Lt//100],axis=0), np.std(data[:10*Lt//100],axis=0)
    return data, mean, std


def save_raw_silence(args):
    data, mean, std = read_raw_silence(args)
    Lt, Ld = data.shape
    path = Path(__file__).parent.resolve()/('../../data/Dt_'+str(args.dt))
    Path(path).mkdir(parents=True, exist_ok=True)
    with h5py.File(path/'silence_train.h5','w') as f:
        for i in range(Ld):
            f.create_dataset('dist'+str(i), data=data[:60*Lt//100,i])
    with h5py.File(path/'silence_validation.h5','w') as f:
        for i in range(Ld):
            f.create_dataset('dist'+str(i), data=data[60*Lt//100:80*Lt//100,i])
    with h5py.File(path/'silence_stats.h5','w') as f:
        for i in range(Ld):
            f.create_dataset('dist'+str(i), data=data[80*Lt//100:,i])
    with h5py.File(path/'mean_std.h5','w') as f:
        f.create_dataset('mean', data=mean)
        f.create_dataset('std', data=std)


def read_raw_events(args):
    sizeofint16 = 2
    filename = select_datafile(args.id, args.data_path)
    Lt, Ld = read_info(filename)
    Lt = Lt//args.dt
    data = np.zeros((Lt,Ld),dtype=np.int16)
    with open(filename,'rb') as f:  
        for j in range(Lt):
            data[j] = np.frombuffer(f.read(Ld*sizeofint16), dtype=np.int16)
            f.seek(Ld*(args.dt-1)*sizeofint16,1)
    return data


def save_raw_events(args):
    data = read_raw_events(args)
    Ld = data.shape[1]
    path = Path(__file__).parent.resolve()/('../../data/Dt_'+str(args.dt))
    Path(path).mkdir(parents=True, exist_ok=True)
    with h5py.File(path/(args.id+'.h5'),'w') as f:
        for i in range(Ld):
            f.create_dataset('dist'+str(i), data=data[:,i])
    f.close()


def save_raw_data(args):
    save_raw_silence(args) if args.id=='silence' else save_raw_events(args)