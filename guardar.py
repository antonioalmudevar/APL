import sys
import pickle
from funciones import binary2matrix

filename = sys.argv[1]
Dt = (int)(sys.argv[2])
B = binary2matrix(filename,Dt)
with open('./data_pkl/'+filename+'_'+str(Dt)+'.pkl', 'wb') as f:
    pickle.dump(B, f)