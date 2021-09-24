import sys
import numpy as np
import pickle
import itertools
from VAE_model import create_encoder, create_decoder
from funciones import binary2matrix, matrix2windows, decisor_vec, decisor_mat, readParam, create_template, expand_decid_kl, evaluate
from time import time

def leer(Dt,filename):
    with open('./data_pkl/'+filename+'_'+str(Dt)+'.pkl','rb') as f:
        B = pickle.load(f)
    return B
    
    
def decisor(breaks,Dt,Nt,Mt,Nd,Md,B):
    decid_kl = np.zeros((B.shape[0],B.shape[1]))
    decid_rec = np.zeros((B.shape[0],B.shape[1]))
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        with open('./param/VAE_param'+formato+'.pkl','rb') as f:
            [param_vec,param_mat] = pickle.load(f)
        Bwind,Lt,Ld = matrix2windows(B[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md)
        Bwind = Bwind[:,:,:,np.newaxis]
        
        encoder = create_encoder(Nt,Nd)
        decoder = create_decoder(Nt,Nd)
        encoder.load_weights('./models/VAE_encoder'+formato+'.h5')
        decoder.load_weights('./models/VAE_decoder'+formato+'.h5')
        
        z_mean, z_log_var, z = encoder.predict(Bwind)
        s_kl = np.sum((z_mean**2 + np.exp(z_log_var) - z_log_var - 1),axis=1)
        d_kl = 1-decisor_vec(s_kl,Lt,Ld,Mt,Md,param_vec)
        decid_kl[:,breaks[i]:breaks[i+1]] = expand_decid_kl(d_kl,Lt,Mt,Ld,Md)
        
        B_mean, B_log_var = decoder.predict(z)
        s_rec = (B_mean - Bwind)**2#/np.exp(B_log_var)
        decid_rec[:,breaks[i]:breaks[i+1]] = decisor_mat(s_rec,Lt,Ld,Nt,Nd,Mt,Md,param_mat)
        
        if(i==0):    
            with open('s_kl.npy', 'wb') as f:
                np.save(f, s_kl)
            with open('s_rec.npy', 'wb') as f:
                np.save(f, s_rec)   
        
    return decid_kl, decid_rec
    
    
paramfile = '/extra/scratch05/aalmudevar/APL/'+sys.argv[1]+'.txt'
filename = sys.argv[2]
breaks, Dt_vec, Nt_vec, Mt_vec, Nd_vec, Md_vec = readParam(paramfile)
for Dt in Dt_vec:
    B = leer(Dt,filename)
    print('Lectura de ficheros - Completado')
    for Nt,Mt_per,Nd,Md_per in itertools.product(Nt_vec,Mt_vec,Nd_vec,Md_vec):
        Mt = int(Mt_per*Nt/100)
        Md = int(Md_per*Nd/100)
        t1 = time()
        decid_kl, decid_rec = decisor(breaks,Dt,Nt,Mt,Nd,Md,B)
        t2 = time()
        auc_kl = evaluate(filename,Dt,decid_kl)
        auc_rec = evaluate(filename,Dt,decid_rec)
        print('Dt = '+str(Dt)+', Nt = '+str(Nt)+', Mt = '+str(Mt)+', Nd = '+str(Nd)+', Md = '+str(Md))
        print(auc_rec)
        print(auc_kl)