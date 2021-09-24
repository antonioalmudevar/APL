import sys
import numpy as np
import pickle
import itertools
from tensorflow.keras.optimizers import Adam
from AE_model import create_model
from funciones import binary2matrix, matrix2windows, param_estad_mat, readParam, select_epochs_bach_AE

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def leer(Dt):
    with open('./data_pkl/silencio1_'+str(Dt)+'.pkl','rb') as f:
        A1 = pickle.load(f)
    with open('./data_pkl/silencio2_'+str(Dt)+'.pkl','rb') as f:
        A2 = pickle.load(f)
    with open('./data_pkl/silencio3_'+str(Dt)+'.pkl','rb') as f:
        B = pickle.load(f)

    A = np.concatenate((A1,A2))
    return(A,B)
  
def entrenar(breaks,nepochs,tam_lote,Dt,Nt,Mt,Nd,Md,A):
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        Awind, _, _ = matrix2windows(A[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md)
        adam = Adam(lr=0.0001)
        model = create_model(Nt,Nd)
        model.compile(optimizer=adam, loss='mean_squared_error')
        Awind = Awind[:,:,:,np.newaxis]
        model.fit(Awind, Awind, epochs=nepochs[i], batch_size=tam_lote[i])
        model.save_weights('./models/AE_model'+formato+'.h5') 
    
def est_param(breaks,Dt,Nt,Mt,Nd,Md,B):
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        model = create_model(Nt,Nd)
        model.load_weights('./models/AE_model'+formato+'.h5')
        Bwind,Lt,Ld = matrix2windows(B[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md) 
        Bwind = Bwind[:,:,:,np.newaxis]
        Bpred = model.predict(Bwind)
        e_rec = (Bwind-Bpred)**2
        param = param_estad_mat(e_rec,Lt,Ld,Nt,Nd,Mt,Md)
        with open('./param/AE_param'+formato+'.pkl', 'wb') as f:
            pickle.dump([param], f)


paramfile = '/extra/scratch05/aalmudevar/APL/'+sys.argv[1]+'.txt'
print('Fichero de parámetros: '+paramfile)
breaks, Dt_vec, Nt_vec, Mt_vec, Nd_vec, Md_vec = readParam(paramfile)
for Dt in Dt_vec:
    A,B = leer(Dt)
    print('Lectura de ficheros - Completado')
    nepochs, tam_lote = select_epochs_bach_AE(Dt)
    for Nt,Mt_per,Nd,Md_per in itertools.product(Nt_vec,Mt_vec,Nd_vec,Md_vec):
        Mt = int(Mt_per*Nt/100)
        Md = int(Md_per*Nd/100)
        print('\n Dt = '+str(Dt)+'\n Nt = '+str(Nt)+'\n Mt = '+str(Mt)+'\n Nd = '+str(Nd)+'\n Md = '+str(Md))
        entrenar(breaks,nepochs,tam_lote,Dt,Nt,Mt,Nd,Md,A)
        print('Entrenamiento de la red - Completado')
        est_param(breaks,Dt,Nt,Mt,Nd,Md,B)
        print('Estimación de parámetros estadísticos - Completado')
