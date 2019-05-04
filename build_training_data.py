import multiprocessing
from functools import partial
from readtraininginput import *
from feature import *
from feature import RCUTOFF_DEFAULT
from feature import nfeature_3
import numpy as np
import time

Rcut = RCUTOFF_DEFAULT
Rcutsq=Rcut*Rcut
cellsize = np.array([Rcut,Rcut,Rcut],dtype='float32')
print('Cutoff Distance and its square',Rcut,Rcutsq,nfeature_3)
nprocessor = 8
ichunk = 45   #64
pool = multiprocessing.Pool(processes=nprocessor)

def construct_feature(filename):
    mc = np.empty([3], dtype='int')
    Natoms, boxsize, atype, position, force = read_training_data(filename)
    print('boxsize:',boxsize)
    cellsize = np.array([Rcut, Rcut, Rcut], dtype='float32')
    for val in range(3):
        mc[val] = int(boxsize[val]/Rcut)
        cellsize[val] = boxsize[val] / float(mc[val])
    if np.min(mc) <= 2:
        Neighbor = makeneighbourlist_0(position, boxsize, Natoms, Rcutsq)
    else:
        lshd, llst, nlist = makelinkedlist(Natoms, boxsize, cellsize, position)
        Neighbor = makeneighbourlist(lshd, llst, position, boxsize, Natoms, cellsize, nlist, Rcutsq)
    halfboxsize = 0.5* boxsize
    print('boxsize 1:',boxsize,halfboxsize)
    prod_x = partial(radial_feature_parallel,
                     Nlocal = ichunk,pos = position,a_type = atype, boxsize = boxsize, 
                     halfboxsize = halfboxsize, Nlist = Neighbor)
    result = pool.imap(prod_x, range(0, Natoms, ichunk), 1)
    for count,val in enumerate(result):
        data = val[0]
        if count == 0:
            tot_feat = data.shape[1]
            atom_features = np.zeros((Natoms,tot_feat), dtype='float32')
#        print(data.shape,data.shape[1],val[1],type(data))
        atom_features[val[1]:val[1]+ichunk,:] = data[:,:]
    print('boxsize 2:',boxsize,halfboxsize)
    print(tot_feat,atom_features.shape,force.shape)
    del Neighbor
    return Natoms,tot_feat,atype,position,force,atom_features


#------xyz to feature vector for NN training-----------
nfiles = 56
filename = 'Train_Al2O3_300/'
for val in range(nfiles):
    f_name=filename+str((val+1)*1)+'.xyz'
    print('reading file: ',f_name)
    t1 = int(round(time.time() * 1000))
    Natoms,nfeature_1,atype,position,property_val,features=construct_feature(f_name)
    t2 = int(round(time.time() * 1000))
    print('time :',t2-t1)
    if val == 0:
        features_tot = features
        property_tot = property_val
        position_tot = position
        Natoms_tot  = Natoms
        atype_tot = atype
    else:
        features_tot  = np.concatenate((features_tot,features),axis=0)
        property_tot  = np.concatenate((property_tot,property_val),axis=0)
        position_tot  = np.concatenate((position_tot,position),axis=0)
        atype_tot =  np.concatenate((atype_tot,atype),axis=0)
        Natoms_tot += Natoms

pool.close()
pool.join()
print("Natoms Nfeature",Natoms_tot,nfeature_1)
print(features_tot.shape)
np.save("DATA/train_XX",features_tot)
np.save("DATA/train_YY",property_tot)
np.save("DATA/train_atype",atype_tot)
np.save("DATA/train_pos",position_tot)

