import sys
import numpy as np
import pickle
import itertools
from tensorflow.keras.optimizers import Adam
from VAE_model import create_encoder, create_decoder, VAE
from funciones import binary2matrix, matrix2windows, param_estad_vec, param_estad_mat, readParam, select_epochs_bach_VAE

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
        adam = Adam(lr=0.001)
        encoder = create_encoder(Nt,Nd)
        decoder = create_decoder(Nt,Nd)
        model = VAE(encoder,decoder)
        model.compile(optimizer=adam)
        Awind = Awind[:,:,:,np.newaxis]
        model.fit(Awind, epochs=nepochs[i],batch_size=tam_lote[i])
        model.encoder.save_weights('./models/VAE_encoder'+formato+'.h5') 
        model.decoder.save_weights('./models/VAE_decoder'+formato+'.h5') 
    

def est_param(breaks,Dt,Nt,Mt,Nd,Md,B):
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        encoder = create_encoder(Nt,Nd)
        decoder = create_decoder(Nt,Nd)
        encoder.load_weights('./models/VAE_encoder'+formato+'.h5')
        decoder.load_weights('./models/VAE_decoder'+formato+'.h5')
        Bwind,Lt,Ld = matrix2windows(B[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md) 
        Bwind = Bwind[:,:,:,np.newaxis]
        z_mean, z_log_var, z = encoder.predict(Bwind)
        B_mean, B_log_var = decoder.predict(z)
        n_kl = np.sum((z_mean**2 + np.exp(z_log_var) - z_log_var - 1),axis=1)
        n_rec = (B_mean - Bwind)**2
        param_vec = param_estad_vec(n_kl,Ld,Md)
        param_mat = param_estad_mat(n_rec,Lt,Ld,Nt,Nd,Mt,Md)
        with open('./param/VAE_param'+formato+'.pkl', 'wb') as f:
            pickle.dump([param_vec,param_mat], f) 
        if(i==0):    
            with open('n_kl.npy', 'wb') as f:
                np.save(f, n_kl)
            with open('n_rec.npy', 'wb') as f:
                np.save(f, n_rec)           

paramfile = '/extra/scratch05/aalmudevar/APL/'+sys.argv[1]+'.txt'
print('Fichero de parámetros: '+paramfile)
breaks, Dt_vec, Nt_vec, Mt_vec, Nd_vec, Md_vec = readParam(paramfile)
for Dt in Dt_vec:
    A,B = leer(Dt)
    print('Lectura de ficheros - Completado')
    nepochs, tam_lote = select_epochs_bach_VAE(Dt)
    for Nt,Mt_per,Nd,Md_per in itertools.product(Nt_vec,Mt_vec,Nd_vec,Md_vec):
        Mt = int(Mt_per*Nt/100)
        Md = int(Md_per*Nd/100)
        print('\n Dt = '+str(Dt)+'\n Nt = '+str(Nt)+'\n Mt = '+str(Mt)+'\n Nd = '+str(Nd)+'\n Md = '+str(Md))
        entrenar(breaks,nepochs,tam_lote,Dt,Nt,Mt,Nd,Md,A)
        print('Entrenamiento de la red - Completado')
        est_param(breaks,Dt,Nt,Mt,Nd,Md,B)
        print('Estimación de parámetros estadísticos - Completado')   

