import sys
import numpy as np
import pickle
import itertools
from AE_LSTM_model import create_model
from funciones import binary2matrix, matrix2groups, groups2windows, decisor_mat, readParam, evaluate
from time import time

def leer(Dt,filename):
    with open('./data_pkl/'+filename+'_'+str(Dt)+'.pkl','rb') as f:
        B = pickle.load(f)
    return B
    
def decisor(breaks,Dt,Nt,Mt,Nd,Md,B,Ng=10,Mg=10):
    decid = np.zeros((B.shape[0],B.shape[1]))
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        Bgroup,Lt,Ld = matrix2groups(B[:,breaks[i]:breaks[i+1]],Ng,Mg,Nt,Mt,Nd,Md) 
        model = create_model(Ng,Nt,Nd)
        model.load_weights('./models/AE_LSTM_model'+formato+'.h5')
        Bpred = model.predict(Bgroup)[:,:,:,:,0]
        e_rec = (Bgroup-Bpred)**2
        e_rec = groups2windows(e_rec,Mg,Mt,Lt,Md,Ld)
        with open('./param/AE_LSTM_param'+formato+'.pkl','rb') as f:
            [param] = pickle.load(f)
        decid[:,breaks[i]:breaks[i+1]] = decisor_mat(e_rec,Lt,Ld,Nt,Nd,Mt,Md,param)
    
    return decid
        
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
        decid = decisor(breaks,Dt,Nt,Mt,Nd,Md,B)
        t2 = time()
        auc = evaluate(filename,Dt,decid)
        print('Dt = '+str(Dt)+', Nt = '+str(Nt)+', Mt = '+str(Mt)+', Nd = '+str(Nd)+', Md = '+str(Md))
        print('\n'+str(auc)+'\n')
        print("Tiempo = %.6f\n\n" % round(t2-t1, 6))