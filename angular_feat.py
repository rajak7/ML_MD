import numpy as np
import math
from feature import  cal_distance
from feature import RCUTOFF_DEFAULT

A_LAMDA=[-1,1]
A_ZETA=[2.0,4.0,6.0]
A_ETA = [2.0,0.5,0.2,0.05]
A_MU = [2.0,2.5,3.0,3.5,4.0]
A_LAMDA_LEN = len(A_LAMDA)
ZETA_LEN = len(A_ZETA)
ETA_LEN = len(A_ETA)
MU_LEN = len(A_MU)

Rcut = RCUTOFF_DEFAULT
Rcutsq=Rcut*Rcut

RC_ANGULAR = 4.0
RC_Ang_DAMP = 8.0

a_nfeature = A_LAMDA_LEN * ZETA_LEN * ETA_LEN * MU_LEN
a_nfeature_3 = 3 * a_nfeature
print('angular feature: ',a_nfeature,a_nfeature_3)

def angular_feature_parallel(start_id,Nlocal,pos,a_type,boxsize,halfboxsize,Nlist):
    features = np.zeros((Nlocal, a_nfeature_3), dtype='float32')
    my_atomid = start_id
    rij_scaled = [0.0,0.0,0.0]
    rik_scaled = [0.0,0.0,0.0]
    G_1 = [0.0,0.0,0.0]
    G_2=[0.0,0.0,0.0]
    for ii in range(Nlocal):
        iatom = start_id + ii
        bond_list = list()
        for jatom in Nlist[iatom,1:Nlist[iatom,0]+1]:
            distance, rij = cal_distance(iatom, jatom, pos, halfboxsize, boxsize)
            if (distance <= RC_ANGULAR and a_type[iatom] != a_type[jatom]):
                bond_list.append(jatom)
        N_angle = len(bond_list)
        for jj in range(N_angle-1):
            jatom = bond_list[jj]
            distance_ij, r_ij = cal_distance(iatom, jatom, pos, halfboxsize, boxsize)
            RIJ_DAMP = 0.5 * (1.0 + math.cos(3.14 * distance_ij / RC_Ang_DAMP))
            rij_scaled[0] = r_ij[0]/distance_ij
            rij_scaled[1] = r_ij[1]/distance_ij
            rij_scaled[2] = r_ij[2]/distance_ij
            for kk in range(jj+1,N_angle):
                katom = bond_list[kk]
                distance_ik, r_ik = cal_distance(iatom, katom, pos, halfboxsize, boxsize)
                RIK_DAMP = 0.5 * (1.0 + math.cos(3.14 * distance_ik / RC_Ang_DAMP))
                rik_scaled[0] = r_ik[0] / distance_ik
                rik_scaled[1] = r_ik[1] / distance_ik
                rik_scaled[2] = r_ik[2] / distance_ik
                rijk_inv = 1.0 / (distance_ij * distance_ik)
                theta_ijk = np.dot(r_ij, r_ik) * rijk_inv
                temp1 = distance_ij - distance_ik * theta_ijk
                temp2 = distance_ik - distance_ij * theta_ijk
                G_1[0] = rij_scaled[0] * temp1 + rik_scaled[0] * temp2
                G_1[1] = rij_scaled[1] * temp1 + rik_scaled[1] * temp2
                G_1[2] = rij_scaled[2] * temp1 + rik_scaled[2] * temp2
                G_1[0] = G_1[0] * rijk_inv * RIJ_DAMP * RIK_DAMP
                G_1[1] = G_1[1] * rijk_inv * RIJ_DAMP * RIK_DAMP
                G_1[2] = G_1[2] * rijk_inv * RIJ_DAMP * RIK_DAMP
                G_2[0] = (rij_scaled[0] + rik_scaled[0]) * RIJ_DAMP * RIK_DAMP
                G_2[1] = (rij_scaled[1] + rik_scaled[1]) * RIJ_DAMP * RIK_DAMP
                G_2[2] = (rij_scaled[2] + rik_scaled[2]) * RIJ_DAMP * RIK_DAMP
                count = 0
                for muval in A_MU:
                    dis2_ij_mu = (distance_ij - muval) * (distance_ij - muval)
                    dis2_ik_mu = (distance_ik - muval) * (distance_ik - muval)
                    for lval in A_LAMDA:
                        cos_l = 1.0+lval*theta_ijk
                        for zval in A_ZETA:
                            cos_l_m_1 = cos_l**(zval-1)
                            cos_l_m = cos_l_m_1 * cos_l
                            const_fact = 2**(1-zval)
                            ga_fact = zval * lval * const_fact * cos_l_m_1
                            gb_fact0 = const_fact * cos_l_m
                            for e_val in A_ETA:
                                gb_fact = (-2.0 * e_val)*gb_fact0
                                dis_fact = math.exp(-1.0*e_val*dis2_ij_mu) * math.exp(-1.0*e_val*dis2_ik_mu)
                                features[ii][count             ] += (G_1[0]*dis_fact*ga_fact + G_2[0]*dis_fact*gb_fact)
                                features[ii][count+1*a_nfeature] += (G_1[1]*dis_fact*ga_fact + G_2[1]*dis_fact*gb_fact)
                                features[ii][count+2*a_nfeature] += (G_1[2]*dis_fact*ga_fact + G_2[2]*dis_fact*gb_fact)
                                count +=1
    return [features,my_atomid]
