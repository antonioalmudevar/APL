import sys
import numpy as np
import pickle
import itertools
from tensorflow.keras.optimizers import Adam
from GAN_model import create_generator, create_feature_extractor, create_discriminator, generator_trainer, get_data_generator, loss
from funciones import binary2matrix, matrix2windows, param_estad_vec, param_estad_mat, readParam, select_epochs_batch_GAN

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
    
    
def entrenar(breaks,Dt,Nt,Mt,Nd,Md,A):
    nepochs, tam_lote = select_epochs_batch_GAN(Dt)
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        Awind, _, _ = matrix2windows(A[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md)
        Awind = Awind[:,:,:,np.newaxis]
        
        generator = create_generator(Nt,Nd)
        feature_extractor = create_feature_extractor(Nt,Nd)
        gan_trainer = generator_trainer(generator,feature_extractor,Nt,Nd)
        losses = {'adv_loss': loss,'cnt_loss': loss}
        lossWeights = {'adv_loss': 1.0, 'cnt_loss': 50.0}
        gan_trainer.compile(optimizer = Adam(learning_rate=1e-3), loss=losses, loss_weights=lossWeights)
        d = create_discriminator(feature_extractor,Nt,Nd)
        d.compile(optimizer=Adam(learning_rate=5e-4), loss='binary_crossentropy')
        
        train_data_generator = get_data_generator(Awind, tam_lote[i])
        for j in range(nepochs[i]):
            x, y = train_data_generator.__next__()
            d.trainable = True
            fake_x = generator.predict(x)
            d_x = np.concatenate([x, fake_x], axis=0)
            d_y = np.concatenate([np.ones(len(x)), np.zeros(len(fake_x))], axis=0)
            d_loss = d.train_on_batch(d_x, d_y)
            d.trainable = False   
            gan_trainer.trainable = True     
            g_loss = gan_trainer.train_on_batch(x, y)
            gan_trainer.trainable = False
            if j % 100 == 0:
                print(f'niter: {j+1}, g_loss: {g_loss}, d_loss: {d_loss}')

        generator.save_weights('./models/GAN_generator'+formato+'.h5')
        feature_extractor.save_weights('./models/GAN_feature_extractor'+formato+'.h5')
        d.save_weights('./models/GAN_discriminator'+formato+'.h5')

        
def est_param(breaks,Dt,Nt,Mt,Nd,Md,B):
    for i in range(len(breaks)-1):
        formato = str(i+1)+'_'+str(Dt)+'_'+str(Nt)+'_'+str(Mt)+'_'+str(Nd)+'_'+str(Md)
        generator = create_generator(Nt,Nd)
        feature_extractor = create_feature_extractor(Nt,Nd)
        generator.load_weights('./models/GAN_generator'+formato+'.h5')
        feature_extractor.load_weights('./models/GAN_feature_extractor'+formato+'.h5')
        Bwind,Lt,Ld = matrix2windows(B[:,breaks[i]:breaks[i+1]],Nt,Mt,Nd,Md) 
        
        Bwind = Bwind[:,:,:,np.newaxis]
        B_pred = generator.predict(Bwind)
        n_rec = (Bwind-B_pred)**2
        param_mat = param_estad_mat(n_rec,Lt,Ld,Nt,Nd,Mt,Md,low_rec=90)
        with open('./param/GAN_param'+formato+'.pkl', 'wb') as f:
            pickle.dump([param_mat], f)


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
        entrenar(breaks,Dt,Nt,Mt,Nd,Md,A)
        print('Entrenamiento de la red - Completado')
        est_param(breaks,Dt,Nt,Mt,Nd,Md,B)
        print('Estimación de parámetros estadísticos - Completado')   
   
