import sys
import numpy as np
import pickle
import itertools
from AE_model import create_model
from funciones import binary2matrix, matrix2windows, decisor_mat, readParam, evaluate, windows2matrix
from time import time, sleep

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def leer(Dt,filename):
    with open('./data_pkl/'+filename+'_'+str(Dt)+'.pkl','rb') as f:
        B = pickle.load(f)
    return B
    
    
def decisor(breaks,Dt,Nt,Mt,Nd,Md,B):
    decid = np.zeros((B.shape[0],B.shape[1]))
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        Bwind,Lt,Ld = matrix2windows(B[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md)
        model = create_model(Nt,Nd)
        model.load_weights('./models/AE_model'+formato+'.h5')
        Bpred = model.predict(Bwind)[:,:,:,0]
        e_rec = (Bwind-Bpred)**2
        with open('./param/AE_param'+formato+'.pkl','rb') as f:
            [param] = pickle.load(f)
        decid[:,breaks[i]:breaks[i+1]] = decisor_mat(e_rec,Lt,Ld,Nt,Nd,Mt,Md,param)
        
    return decid
    
        
#Parametros de entrada
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
        print('Dt = '+str(Dt)+', Nt = '+str(Nt)+', Mt = '+str(Mt)+', Nd = '+str(Nd)+', Md = '+str(Md)+'\n')
        print(auc)
        #with open("AE_results_"+str(Dt)+".txt", "a") as f:
            #f.write('Dt = '+str(Dt)+', Nt = '+str(Nt)+', Mt = '+str(Mt)+', Nd = '+str(Nd)+', Md = '+str(Md)+'\n')
            #f.write("%.6f\n" %round(sum(auc)/3,6))
