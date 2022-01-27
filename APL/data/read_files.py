def select_datafile(id_file, data_path):
    if id_file=='silencio1':
        filename_binary = data_path+'/20200214_DAS010_silencio1'
    elif id_file=='silencio2':
        filename_binary = data_path+'/20200214_DAS010_silencio2'
    elif id_file=='silencio3':
        filename_binary = data_path+'/20200214_DAS010_silencio3'
    elif id_file=='entra':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraEntra'
    elif id_file=='oruga0m':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_0m'
    elif id_file=='oruga5m':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_5m'
    elif id_file=='oruga10m':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_10m'
    elif id_file=='cazo0m':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_cazo0m'
    elif id_file=='cazo5m':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_cazo5m'
    elif id_file=='cazo10m':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_cazo10m'
    elif id_file=='hidraulico1':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_MartilloHidraulico_0_5_10m_iteracion1'  
    elif id_file=='hidraulico2':
        filename_binary = data_path+'/DAS010/20200212_DAS010_ExcavadoraOruga_MartilloHidraulico_0_5_10m_iteracion2'  
    return filename_binary


def read_info(filename):
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
    Lt = (int)(info['T']*info['Fp'])
    Ld = (int)(info['Ns'])
    return Lt, Ld