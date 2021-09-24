# DECLARACIÃ“N DE FUNCIONES UTILIZADAS

import numpy as np
import csv

# FUNCIONES PARA LA SELECCIÓN DE FICHEROS
def select_binary(nfile):
    rem = False
    folder = '/extra/scratch05/database06/AragonPhotonics/20200212Betancourt/20200212 RAWS EINA/'
    if nfile=='silencio1':
        filename_binary = folder+'20200214_DAS010_silencio1'
        rem = True
    elif nfile=='silencio2':
        filename_binary = folder+'20200214_DAS010_silencio2'
        rem = True
    elif nfile=='silencio3':
        filename_binary = folder+'20200214_DAS010_silencio3'
        rem = True
    elif nfile=='entra':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraEntra'
    elif nfile=='oruga0m':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_0m'
    elif nfile=='oruga5m':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_5m'
    elif nfile=='oruga10m':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_10m'
    elif nfile=='cazo0m':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_cazo0m'
    elif nfile=='cazo5m':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_cazo5m'
    elif nfile=='cazo10m':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_cazo10m'
    elif nfile=='hidraulico1':
        filename_binary = folder+'DAS010/20200212_DAS010_ExcavadoraOruga_MartilloHidraulico_0_5_10m_iteracion1'  
    
    return filename_binary, rem


# FUNCIONES PARA LECTURA DE DATOS Y DE CABECERA
def bytes2int16 (x):
    sizeofint16 = 2
    L = len(x)
    result = np.zeros((int)(L/2),np.int16)
    for i in range(0,L,sizeofint16):
        result[i//2] = int.from_bytes(x[i:i+sizeofint16],'little',signed=False) 
        
    return result

def readInfo (filename):
    infoID = open(filename+'_info.txt','r')
    for line in infoID:
        index = line.find(':')
        param = line[0:index] 
        if(param=="Muestras por traza"):
            Ns = int(line[index+2:-1])
        elif(param=="Muestras por trama"):
            Ns = int(line[index+2:-1])
        elif(param=="Frecuencia de trigger en Hz"):
            Fp = int(line[index+2:-1])
        elif(param=="Frecuencia de repeticion de pulso"):
            Fp = int(line[index+2:-4])
        elif(param=="Tiempo de guardado en segundos"):  
            T = int(line[index+2:-1])
        elif(param=="Segundos"):  
            T = int(line[index+2:-1])     
    infoID.close()
    info = {
        'vluz' : 2e8,
        'Fs' : 125e6,
        'Ds2' : 8,
        'Dt' : 8,
        'Ns' : Ns,
        'Fp' : Fp,
        'T' : T}
    
    return info

def readParam(paramfile):
    
    with open(paramfile) as f:
        reader = csv.reader(f)
        data = list(reader)
        breaks = list(map(int, data[0]))
        Dt = list(map(int, data[1]))
        Nt = list(map(int, data[2]))
        Mt = list(map(int, data[3]))
        Nd = list(map(int, data[4]))
        Md = list(map(int, data[5]))
        
        
    return(breaks,Dt,Nt,Mt,Nd,Md)

def binary2matrix (nfile, Dt):
    filename,rem = select_binary(nfile)
    info = readInfo(filename)
    ndist = (int)(info['Ns'])
    ntiempo = (int)(info['T']*info['Fp']/(Dt))
    sizeofint16 = 2
    fileID = open(filename,'rb')
    byte_list = []
    for i in range(ntiempo):
        #print('%.2f%%' % ((i/ntiempo)*100))
        byte_list.append(bytes2int16(fileID.read(ndist*sizeofint16)))
        fileID.seek(ndist*(Dt-1)*sizeofint16,1)
    fileID.close()
    byte_str = np.asarray(byte_list)
    A = byte_str.reshape(ntiempo,ndist).astype(np.float16)
    if rem:
        A = A[:,:-440]
    
    return A


# FUNCIONES PARA PROCESADO DE DATOS
def minmax_norm (X):
    Xnorm = (X-X.min())/(X.max()-X.min())
    return Xnorm    


def enventanar(x,N,M):
    T = len(x)
    m = np.arange(0,T-N+1,M)
    L = len(m)
    ind = np.dot((np.arange(0,N)).reshape(N,1),np.ones((1,L))) + np.dot(np.ones((N,1)),m.reshape(1,L))
    X = np.swapaxes(x[ind.astype(int)],0,1)
    pad = (int)(np.ceil(T/M)-L)
    if pad>0:
        X = np.repeat(X, [1]*(X.shape[0]-1)+[(int)(pad+1)], axis=0)
    return X


def desenventanar(X,M,L): 
    x = X[:,:M].reshape(-1, *X.shape[2:])
    x = x[:L]
    return x


def matrix2windows(A,Nt,Mt,Nd,Md,norm=True):
    Lt = A.shape[0]
    Ld = A.shape[1]
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    # Enventanado en tiempo
    B = enventanar(A,Nt,Mt) 
    # Enventanado en distancia
    C = (enventanar(B.T,Nd,Md).T).swapaxes(2,3)
    D = np.swapaxes(C,1,2)
    Awind = D.reshape(-1,Nt,Nd)
    if (norm):
        for i in range(ntiempo*ndist):
            Awind[i] = minmax_norm(Awind[i])
    
    return Awind,Lt,Ld

def windows2groups(Awind,Ng,Mg,Mt,Lt,Md,Ld):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    Nt = Awind.shape[1]
    Nd = Awind.shape[2]
    
    B = Awind.reshape(ntiempo,ndist,Nt,Nd)
    C = enventanar(B,Ng,Mg)
    D = np.swapaxes(C,1,2)
    Agroup = D.reshape(-1,Ng,Nt,Nd)
    
    return Agroup

def matrix2groups(A,Ng,Mg,Nt,Mt,Nd,Md,norm=True):
    Awind,Lt,Ld = matrix2windows(A,Nt,Mt,Nd,Md,norm=True)
    Agroup = windows2groups(Awind,Ng,Mg,Mt,Lt,Md,Ld)

    return Agroup, Lt, Ld


def windows2matrix(Awind,Nt,Mt,Lt,Nd,Md,Ld):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    D = Awind.reshape(ntiempo,ndist,Nt,Nd)
    C = np.swapaxes(D,1,2)
    # Desenventanado en distancia
    B = np.zeros((ntiempo,Nt,Ld),np.float16)
    for i in range(ntiempo):
        for j in range(Nt):
            B[i,j] = desenventanar(C[i,j],Md,Ld)
    A = np.zeros((Lt,Ld),np.float16)
    # Desenventanado en tiempo
    for i in range(Ld):
        A[:,i] = desenventanar(B[:,:,i],Mt,Lt)
        
    return A

def groups2windows(Agroup,Mg,Mt,Lt,Md,Ld):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    Ng = Agroup.shape[1]
    Nt = Agroup.shape[2]
    Nd = Agroup.shape[3]

    D = Agroup.reshape(-1,ndist,Ng,Nt,Nd)
    C = np.swapaxes(D,1,2)
    B = desenventanar(C,Mg,ntiempo)
    Awind = B.reshape(ndist*ntiempo,Nt,Nd)
    
    return Awind

def groups2matrix (Agroup,Ng,Mg,Nt,Mt,Lt,Nd,Md,Ld):
    Awind = groups2windows(Agroup,Mg,Mt,Lt,Md,Ld)
    A = windows2matrix(Awind,Nt,Mt,Lt,Nd,Md,Ld)
    
    return A

# Modelos divididos por distancia
def divide_windows_bydist(Awind,Ld,ini):
    Nt = Awind.shape[1]
    Nd = Awind.shape[2]
    ndist = Ld//Nd
    ntiempo = Awind.shape[0]//ndist
    fin = np.append(ini[1:],ndist)
    B = Awind.reshape(ntiempo,ndist,Nt,Nd)
    Adivid = []
    for i in range(len(ini)):
        C = (B[:,ini[i]:fin[i]]).reshape(-1,Nt,Nd)
        Adivid.append(C)
    
    return Adivid

def join_windows_bydist(Adivid,Ld,ini):
    Nt = Adivid[0].shape[1]
    Nd = Adivid[0].shape[2]
    ndist = Ld//Nd
    fin = np.append(ini[1:],ndist)
    lon = fin-ini
    ntiempo = Adivid[0].shape[0]//lon[0]
    Awind = np.zeros((ntiempo,ndist,Nt,Nd))
    for i in range(len(ini)):
        B = Adivid[i].reshape(ntiempo,lon[i],Nt,Nd)
        Awind[:,ini[i]:fin[i]] = B
        
    return Awind

def dividwindows2matrix(Adivid,Nt,Mt,Lt,Nd,Md,Ld,ini):
    Awind = join_windows_bydist(Adivid,Ld,ini)
    A = windows2matrix(Awind,Nt,Mt,Lt,Nd,Md,Ld)
    
    return A

def matrix_window_energy(A,Nt,Mt,Nd,Md,Lutil=5560):
    Lt = A.shape[0]
    Ld = A.shape[1]
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    # Enventanado en tiempo
    B = enventanar(A,Nt,Mt) 
    # Enventanado en distancia
    C = (enventanar(B.T,Nd,Md).T).swapaxes(2,3)     
    D = np.swapaxes(C,1,2)
    E = D.reshape(ntiempo*ndist,Nt,Nd)
    F = np.mean(E,axis=(1,2))
    Ewind = F.reshape(ntiempo,ndist)
    Ewind = Ewind[:,:Lutil]

    return Ewind,Lt,Ld
    
# FUNCIONES DE DECISOR
from scipy.stats import expon, norm
def param_estad_vec(n_vec,Ld,Md):
    ndist = (int)(np.ceil(Ld/Md))
    D = n_vec.reshape(-1,ndist).swapaxes(0,1)
    param = np.zeros((ndist,2))
    for i in range(ndist):
        D[i].sort()
        param[i] = norm.fit(D[i,len(D[i])//100:99*len(D[i])//100])
        
    return param

def decisor_vec(s_vec,Lt,Ld,Mt,Md,param):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    D = s_vec.reshape(ntiempo,ndist).swapaxes(0,1)
    prob = np.zeros((ndist,ntiempo))
    param_extend = np.zeros((ndist,2))
    param_extend[:len(param),:] = param
    param_extend[len(param):ndist,:] = param[-1:]
    for i in range(ndist):
        prob[i] = norm.cdf(D[i],param_extend[i,0],param_extend[i,1])
    prob = (prob.swapaxes(0,1))
    
    return prob
    
def param_estad_mat(n_mat,Lt,Ld,Nt,Nd,Mt,Md,low_rec=90):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    D = n_mat.reshape(ntiempo,ndist,Nt,Nd).swapaxes(0,1)
    D = D.reshape(ndist,-1)
    param = np.zeros((ndist,2))
    for i in range(ndist):
        D[i].sort()
        rec_win = D[i,low_rec*ntiempo*Nt*Nd//100:]
        param[i] = expon.fit(rec_win)
        
    return param


def decisor_mat(s_mat,Lt,Ld,Nt,Nd,Mt,Md,param):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    D = s_mat.reshape(ntiempo,ndist,Nt,Nd).swapaxes(0,1)
    prob = np.zeros((ndist,ntiempo,Nt,Nd))
    param_extend = np.zeros((ndist,2))
    param_extend[:len(param),:] = param
    param_extend[len(param):ndist,:] = param[-1:]
    for i in range(ndist):
        prob[i] = expon.cdf(D[i],param_extend[i,0],param_extend[i,1])
    prob = prob.swapaxes(0,1)
    decid = windows2matrix(prob,Nt,Mt,Lt,Nd,Md,Ld)
    
    return decid
    
def expand_decid_kl(decid_kl,Lt,Mt,Ld,Md):
    ntiempo = (int)(np.ceil(Lt/Mt))
    ndist = (int)(np.ceil(Ld/Md))
    decid_kl = decid_kl.reshape(ntiempo,ndist)
    decid_kl_expand = np.zeros((Lt,Ld))
    for i in range(ntiempo):
        for j in range(ndist):
            decid_kl_expand[i*Mt:(i+1)*Mt,j*Md:(j+1)*Md] = decid_kl[i,j]
            
    return decid_kl_expand

def create_template(filename):
    col_idx = np.concatenate([np.arange(3,6), np.arange(104,107), np.arange(301,304), np.arange(690,693)])
    if(filename=='entra'):
        template = np.zeros((300,695))
        row_idx = np.concatenate([np.arange(9,113)])
    elif(filename=='oruga0m'):
        template = np.zeros((240,695))
        row_idx = np.concatenate([np.arange(63,100), np.arange(140,191)])
    elif(filename=='oruga5m'):
        template = np.zeros((240,695))
        row_idx = np.concatenate([np.arange(61,102), np.arange(140,181)])
    elif(filename=='oruga10m'):
        template = np.zeros((240,695))
        row_idx = np.concatenate([np.arange(61,100), np.arange(149,193)])     
    elif(filename=='hidraulico1'):
        template = np.zeros((120,695))
        row_idx = np.concatenate([np.arange(47,56), np.arange(86,94), np.arange(113,120)])
         
    template[row_idx[:, None], col_idx] = 1
    return (template==1)
    
    
from sklearn.metrics import roc_auc_score, roc_curve
def evaluate(filename, Dt, decid):
    template = create_template(filename)
    result = np.zeros((template.shape[0],750))
    for i in range (decid.shape[0]*Dt//1000):
        for j in range (decid.shape[1]//8):
            result[i,j] = np.mean(decid[i*1000//Dt:(i+1)*1000//Dt,j*8:(j+1)*8])      
    result = result[:,:695]
    
    auc = np.zeros((3))
    auc[0] = roc_auc_score(template[:,:250].reshape(-1), result[:,:250].reshape(-1))
    auc[1] = roc_auc_score(template[:,250:500].reshape(-1), result[:,250:500].reshape(-1))
    auc[2] = roc_auc_score(template[:,500:].reshape(-1), result[:,500:].reshape(-1))
    
    fpr_val, tpr_val, _ = roc_curve(template[:,:250].reshape(-1), result[:,:250].reshape(-1))
    np.savez(filename+'1.npz', fpr=fpr_val, tpr=tpr_val)
    fpr_val, tpr_val, _ = roc_curve(template[:,250:500].reshape(-1), result[:,250:500].reshape(-1))
    np.savez(filename+'2.npz', fpr=fpr_val, tpr=tpr_val)
    fpr_val, tpr_val, _ = roc_curve(template[:,500:].reshape(-1), result[:,500:].reshape(-1))
    np.savez(filename+'3.npz', fpr=fpr_val, tpr=tpr_val)
        
    return auc
    
def select_epochs_bach_AE(Dt):
    if (Dt==10):
        return [120,160,200,240], [64,32,16,16]
    elif (Dt==40):
        return [160,200,240,280], [64,32,16,16]
    elif (Dt==100):
        return [200,240,280,320], [64,32,16,16]
        
def select_epochs_batch_AE_LSTM(Dt):
    if (Dt==10):
        return [40,50,90,110], [64,64,32,16]
    elif (Dt==40):
        return [160,200,240,280], [64,32,16,16]
    
def select_epochs_bach_VAE(Dt):
    if (Dt==10):
        return [30,40,60,70], [64,64,32,32]
    elif (Dt==40):
        return [160,200,240,280], [64,32,16,16]
    elif (Dt==100):
        return [60,70,90,100], [64,64,32,32]
        
def select_epochs_batch_GAN(Dt):
    if (Dt==10):
        return [20000,30000,50000,60000], [64,64,32,32]
    elif (Dt==40):
        return [30000,40000,60000,70000], [64,64,32,32]