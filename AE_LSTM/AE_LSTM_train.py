import sys
import numpy as np
import pickle
import itertools
from tensorflow.keras.optimizers import Adam
from AE_LSTM_model import create_model
from funciones import binary2matrix, matrix2groups, groups2windows, param_estad_mat, readParam, select_epochs_batch_AE_LSTM

def leer(Dt):
    with open('./data_pkl/silencio1_'+str(Dt)+'.pkl','rb') as f:
        A1 = pickle.load(f)
    with open('./data_pkl/silencio2_'+str(Dt)+'.pkl','rb') as f:
        A2 = pickle.load(f)
    with open('./data_pkl/silencio3_'+str(Dt)+'.pkl','rb') as f:
        B = pickle.load(f)

    A = np.concatenate((A1,A2))
    return(A,B)
    
def entrenar(breaks,Dt,Ng,Mg,Nt,Mt,Nd,Md,A):
    nepochs, tam_lote = select_epochs_batch_AE_LSTM(Dt)
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        Agroup, _, _ = matrix2groups(A[:,breaks[i]:breaks[i+1]],Ng,Mg,Nt,Mt,Nd,Md)
        model = create_model(Ng,Nt,Nd)
        model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error')
        Agroup = Agroup[:,:,:,:,np.newaxis]
        model.fit(Agroup, Agroup, epochs=nepochs[i], batch_size=tam_lote[i])
        model.save_weights('./models/AE_LSTM_model'+formato+'.h5') 
    
def est_param(breaks,Dt,Ng,Mg,Nt,Mt,Nd,Md,B):
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        model = create_model(Ng,Nt,Nd)
        model.load_weights('./models/AE_LSTM_model'+formato+'.h5')
        Bgroup,Lt,Ld = matrix2groups(B[:,breaks[i]:breaks[i+1]],Ng,Mg,Nt,Mt,Nd,Md) 
        Bgroup = Bgroup[:,:,:,:,np.newaxis]
        Bpred = model.predict(Bgroup)
        e_rec = (Bgroup-Bpred)**2
        e_rec = groups2windows(e_rec,Mg,Mt,Lt,Md,Ld)
        param = param_estad_mat(e_rec,Lt,Ld,Nt,Nd,Mt,Md)
        with open('./param/AE_LSTM_param'+formato+'.pkl', 'wb') as f:
            pickle.dump([param], f)


#Parametros de entrada
print('Entrada al programa')
paramfile = '/extra/scratch05/aalmudevar/APL/'+sys.argv[1]+'.txt'
print('Fichero de parámetros: '+paramfile)
breaks, Dt_vec, Nt_vec, Mt_vec, Nd_vec, Md_vec = readParam(paramfile)
for Dt in Dt_vec:
    A,B = leer(Dt)
    print('Lectura de ficheros - Completado')
    for Nt,Mt_per,Nd,Md_per in itertools.product(Nt_vec,Mt_vec,Nd_vec,Md_vec):
        Mt = int(Mt_per*Nt/100)
        Md = int(Md_per*Nd/100)
        print('\n Dt = '+str(Dt)+'\n Nt = '+str(Nt)+'\n Mt = '+str(Mt)+'\n Nd = '+str(Nd)+'\n Md = '+str(Md))
        entrenar(breaks,Dt,10,10,Nt,Mt,Nd,Md,A)
        print('Entrenamiento de la red - Completado')
        est_param(breaks,Dt,10,10,Nt,Mt,Nd,Md,B)
        print('Estimación de parámetros estadísticos - Completado')
