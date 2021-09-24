import sys
import numpy as np
import pickle
import itertools
from GAN_model import create_generator, create_feature_extractor, create_discriminator
from funciones import binary2matrix, matrix2windows, decisor_vec, decisor_mat, readParam, create_template, evaluate, windows2matrix
from sklearn.metrics import roc_auc_score

def leer(B,filename):
    with open('./data_pkl/'+filename+'_'+str(Dt)+'.pkl', 'rb') as f:
        B = pickle.load(f)
    return B
    
def decisor(breaks,Dt,Nt,Mt,Nd,Md,B):
    decid_rec = np.zeros((B.shape[0],B.shape[1]))
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        with open('./param/GAN_param'+formato+'.pkl','rb') as f:
            [param_mat] = pickle.load(f)
        Bwind,Lt,Ld = matrix2windows(B[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md)
        Bwind = Bwind[:,:,:,np.newaxis]
        
        generator = create_generator(Nt,Nd)
        feature_extractor = create_feature_extractor(Nt,Nd)
        generator.load_weights('./models/GAN_generator'+formato+'.h5')
        feature_extractor.load_weights('./models/GAN_feature_extractor'+formato+'.h5')

        B_pred = generator.predict(Bwind)
        s_rec = (Bwind-B_pred)**2
        decid_rec[:,breaks[i]:breaks[i+1]] = decisor_mat(s_rec,Lt,Ld,Nt,Nd,Mt,Md,param_mat)
        
        ntiempo = (int)(np.ceil(Lt/Mt))
        ndist = (int)(np.ceil(Ld/Md))
        
    return decid_rec
        
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
        decid = decisor(breaks,Dt,Nt,Mt,Nd,Md,B)
        auc = evaluate(filename,Dt,decid)
        print(auc)
