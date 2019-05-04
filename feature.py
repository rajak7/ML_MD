import numpy as np
import math

mu=[1.5,2.0,2.3,2.5,3.0,3.3,3.5,3.7,4.0,4.25,4.5,4.75,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
eta=[0.5,1.0,2.0,3.0,8.0]
mu_list = len(mu)
eta_list = len(eta)

RCUTOFF_DEFAULT = 10.00  #9.50
RCUTOFF_DAMP = 15.0
ATOMS_TYPE = 3

if ATOMS_TYPE == 1:
    nfeature = mu_list * eta_list
elif ATOMS_TYPE == 2:
    nfeature = 2 * (mu_list * eta_list)
    nfeature_1 = (mu_list * eta_list)
elif ATOMS_TYPE == 3:
    nfeature = 3 * (mu_list * eta_list)
    nfeature_1 = (mu_list * eta_list)
else:
    print(ATOMS_TYPE,' is greater then 3')
    exit(1)

nfeature_3 = 3 * nfeature
print('Cutoff Distance: ',RCUTOFF_DEFAULT,RCUTOFF_DAMP,nfeature,nfeature_3)

def radial_feature(Natoms,pos,a_type,boxsize,halfboxsize,Nlist):
    features = np.zeros((Natoms,nfeature_3),dtype='float32')
    for iatom in range(Natoms):
        for jatom in Nlist[iatom,1:Nlist[iatom,0]+1]:
            count = 0
            itag = (a_type[iatom] == a_type[jatom])
            distance, rij = cal_distance(iatom, jatom, pos, halfboxsize, boxsize)
            projection_x = rij[0] / distance
            projection_y = rij[1] / distance
            projection_z = rij[2] / distance
            temp1 = 0.5 * (1.0 + math.cos(3.14 * distance / RCUTOFF_DAMP))
            for muval in mu:
                temp2 = (distance - muval)*(distance - muval)
                for etaval in eta:
                    temp3 = temp1*math.exp(-1.0 * etaval * temp2)
                    if ATOMS_TYPE == 1:
                        features[iatom][count  ] += projection_x * temp3
                        features[iatom][count+nfeature] += projection_y * temp3
                        features[iatom][count+2*nfeature] += projection_z * temp3
                    else:
                        if itag == True:
                            features[iatom][count  ] += projection_x * temp3
                            features[iatom][count+2*nfeature_1] += projection_y * temp3
                            features[iatom][count+4*nfeature_1] += projection_z * temp3
                        else:
                            features[iatom][count+nfeature_1] += projection_x * temp3
                            features[iatom][count+3*nfeature_1] += projection_y * temp3
                            features[iatom][count+5*nfeature_1] += projection_z * temp3
                    count += 1
    return features

def radial_feature_parallel(start_id,Nlocal,pos,a_type,boxsize,halfboxsize,Nlist):
    features = np.zeros((Nlocal,nfeature_3),dtype='float32')
    my_atomid = start_id
    for ii in range(Nlocal):
        iatom = start_id+ii
        for jatom in Nlist[iatom,1:Nlist[iatom,0]+1]:
            count = 0
            itag = (a_type[iatom] == a_type[jatom])
            distance, rij = cal_distance(iatom, jatom, pos, halfboxsize, boxsize)
            projection_x = rij[0] / distance
            projection_y = rij[1] / distance
            projection_z = rij[2] / distance
            temp1 = 0.5 * (1.0 + math.cos(3.14 * distance / RCUTOFF_DAMP))
            for muval in mu:
                temp2 = (distance - muval)*(distance - muval)
                for etaval in eta:
                    temp3 = temp1*math.exp(-1.0 * etaval * temp2)
                    if ATOMS_TYPE == 1:
                        features[ii][count  ] += projection_x * temp3
                        features[ii][count+nfeature] += projection_y * temp3
                        features[ii][count+2*nfeature] += projection_z * temp3
                    else:
                        if itag == True:
                            features[ii][count  ] += projection_x * temp3
                            features[ii][count+3*nfeature_1] += projection_y * temp3
                            features[ii][count+6*nfeature_1] += projection_z * temp3
                        else:
                            ival = True   #for (01), (10) and (21) pairs
                            if a_type[iatom] == 0 and a_type[jatom] == 2:
                                ival = False
                            elif a_type[iatom] == 1 and a_type[jatom] == 2:
                                ival = False
                            elif a_type[iatom] == 2 and a_type[jatom] == 0:
                                ival = False
                            if ival == True:
                                features[ii][count+nfeature_1] += projection_x * temp3
                                features[ii][count+4*nfeature_1] += projection_y * temp3
                                features[ii][count+7*nfeature_1] += projection_z * temp3
                            else:
                                features[ii][count+2*nfeature_1] += projection_x * temp3
                                features[ii][count+5*nfeature_1] += projection_y * temp3
                                features[ii][count+8*nfeature_1] += projection_z * temp3
                    count += 1
    return [features,my_atomid]

def cal_distance(iatom,jatom,positions,halfbox,boxsize):
    rsq = 0.0
    dr=[0.0,0.0,0.0]
    for kk in range(3):
        dr[kk] = positions[iatom][kk] - positions[jatom][kk]
        if (dr[kk] > halfbox[kk]): dr[kk] -= boxsize[kk]
        if (dr[kk] < -1.0*halfbox[kk]): dr[kk] += boxsize[kk]
        rsq += dr[kk] * dr[kk]
    distance = math.sqrt(rsq)
    return distance,dr
