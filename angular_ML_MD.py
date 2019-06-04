import multiprocessing
from functools import partial
from buildneighbor import *
from init_MD import *
from feature import *
from angular_feat import *
from feature import nfeature_3 as r_feat
from angular_feat import a_nfeature_3 as a_feat
from feature import nfeature as r_x_feat
from angular_feat import a_nfeature as a_x_feat
from NN_Training.neural_network import *
import time

nprocessor = 8
ichunk = 48
b_size = 1
feat_dim = r_x_feat + a_x_feat
tot_feat = r_feat + a_feat
network_param={'input_dim':feat_dim,'output_dim':1,'num_epoch':300,
              'batch_size':256,'drop_prob':0.25,'decay_rate':5e-4}
fname = ['Si_x.ckpt','Si_y.ckpt','Si_z.ckpt','C_x.ckpt','C_y.ckpt','C_z.ckpt']
mean_val=['feature_mean_Si.npy','feature_std_Si.npy','feature_mean_C.npy','feature_std_C.npy']

print("Total feature dimension: ",tot_feat,r_feat,a_feat)
print("Per direction feature: ",feat_dim,r_x_feat,a_x_feat)
pool = multiprocessing.Pool(processes=nprocessor)


def load_mean_std():
    norm_const=list()
    for val in mean_val:
        input_name = 'NN_Training/'+val
        tt = np.load(input_name)
        norm_const.append(tt)
    return norm_const


def loadNN_model():
    tf.reset_default_graph()
    sess = list()
    ML_Force_Cal = list()

    for ii in range(len(fname)):
        new_graph = tf.Graph()
        fmodel = create_sess(new_graph, network_param, fname=fname[ii])
        sess_x, ML_Force_Cal_x = fmodel.load_model()
        sess.append(sess_x)
        ML_Force_Cal.append(ML_Force_Cal_x)
    return sess,ML_Force_Cal

def compute_force(sess,ML_Force,atom_feature,atoms,norm_const):
    for ii in range(atoms.localN):
        for val in range(3):
            atom_feat = np.concatenate((atom_feature[ii, val * r_x_feat : (val + 1) * r_x_feat],
                                        atom_feature[ii,r_feat + (val * a_x_feat):r_feat +((val + 1) * a_x_feat)]),
                                        axis=0)
            if atoms.atype[ii] == 0:
                ftemp = ML_Force[val].predict_force_single(sess[val],atom_feat)
                atoms.force[ii, val] = ftemp
                #atoms.force[ii, val] = (ftemp * norm_const[5][val]) + norm_const[4][val]
            elif atoms.atype[ii] == 1:
                ftemp = ML_Force[val+3].predict_force_single(sess[val+3],atom_feat)
                atoms.force[ii, val] = ftemp
                #atoms.force[ii, val] = (ftemp * norm_const[7][val]) + norm_const[6][val]
            elif atoms.atype[ii] == 2:
                ftemp = ML_Force[val+6].predict_force_single(sess[val+6],atom_feat)
                atoms.force[ii, val] = ftemp
            else:
                print("only three components are currently supported")
                exit(1)
    return

def time_verlet(ival,atoms,sess,ML_Force,md_const,md_param,norm_const):
    for i in range(atoms.localN):
        for k in range(3):
            atoms.vel[i,k] += md_const.DTHALF * atoms.force[i,k] * md_const.ACCEL_FAC[atoms.atype[i]]
            atoms.pos[i,k] += atoms.vel[i,k]*md_const.DT
            # Apply PBC Condtion
            if atoms.pos[i,k] < 0.0:
                atoms.pos[i,k] += md_param.boxmd[k]
            elif atoms.pos[i,k] > md_param.boxmd[k]:
                atoms.pos[i,k] -= md_param.boxmd[k]

    #Calculate the forces and update velocities again to complete the step
    if (atoms.mc[0] >= 3):
        buildList(atoms.localN,atoms.NMAX,atoms.pos,atoms.nheader,atoms.lsize,atoms.linklist,atoms.mc,atoms.cellsize)
        neighborlist(atoms,md_param.boxmd,md_param.halfboxmd,md_const)
    else:
        neighborlist_0(atoms,md_param.boxmd,md_param.halfboxmd,md_const)

    atom_feature = compute_feature(md_param, atoms, norm_const)

    compute_force(sess, ML_Force, atom_feature,atoms,norm_const)
    atoms.force /= md_const.f_scale

    for i in range(atoms.localN):
        for k in range(3):
            atoms.vel[i,k] += md_const.DTHALF * atoms.force[i,k] * md_const.ACCEL_FAC[atoms.atype[i]]


    if md_param.iscale_v <= 2 and ( (ival % md_param.iscale_step) == 0) :
        scale_temp(md_param,md_const,atoms,md_param.iscale_v)
    return

def compute_feature(md_param,atoms,norm_const):
    atom_feature = np.zeros((atoms.localN, tot_feat), dtype='float32')
    # radial feature
    prod_r = partial(radial_feature_parallel,
                     Nlocal=ichunk, pos=atoms.pos,a_type = atoms.atype, boxsize=md_param.boxmd,
                     halfboxsize=md_param.halfboxmd, Nlist=atoms.Neighbor)
    result_r = pool.imap(prod_r, range(0, atoms.localN, ichunk), 1)
    # angular feature
    prod_a = partial(angular_feature_parallel,
                     Nlocal=ichunk, pos=atoms.pos, a_type=atoms.atype, boxsize=md_param.boxmd,
                     halfboxsize=md_param.halfboxmd, Nlist=atoms.Neighbor)
    result_a = pool.imap(prod_a, range(0, atoms.localN, ichunk), 1)
    for val_a,val_b in zip(result_r,result_a):
        data1 = val_a[0]
        data2 = val_b[0]
        #print(data1.shape,data2.shape,r_feat,tot_feat,tot_feat-r_feat,atoms.localN)
        atom_feature[val_a[1]:val_a[1] + ichunk, 0:r_feat] = data1[:, :]
        atom_feature[val_b[1]:val_b[1] + ichunk, r_feat:tot_feat] = data2[:, :]
    
    #normalize the feature values
    for ii in range(atoms.localN):
        if atoms.atype[ii] == 0:
            atom_feature[ii,:] = (atom_feature[ii,:] - norm_const[0]) / norm_const[1]
        elif atoms.atype[ii] == 1:
            atom_feature[ii,:] = (atom_feature[ii,:] - norm_const[2]) / norm_const[3]
        elif atoms.atype[ii] == 2:
            atom_feature[ii,:] = (atom_feature[ii,:] - norm_const[4]) / norm_const[5]
        else:
            print("Maximum Three component is supported")
            exit(1)
    return atom_feature

def main():

    force_vel_stat = open('force_val.txt', 'w')
    out_stat =  open('log.txt', 'w')

    md_param, md_const, atoms = initilize()
    if md_const.CUTOFF != RCUTOFF_DEFAULT:
        print("RCUTOFF_DEFAULT Value in feature.py different then md_const.CUTOFF:",md_const.CUTOFF,RCUTOFF_DEFAULT)
        exit(1)

    sess,ML_Force = loadNN_model()
    norm_const = load_mean_std()
    print('number of linked list cells :', atoms.mc)


    t1 = int(round(time.time() * 1000))
    if (atoms.mc[0] >= 3):
        buildList(atoms.localN, atoms.NMAX, atoms.pos, atoms.nheader, atoms.lsize, atoms.linklist, atoms.mc, atoms.cellsize)
        neighborlist(atoms, md_param.boxmd, md_param.halfboxmd, md_const)
    else:
        neighborlist_0(atoms, md_param.boxmd, md_param.halfboxmd, md_const)


    t2 = int(round(time.time() * 1000))
    atom_feature = compute_feature(md_param,atoms,norm_const)
    t3 = int(round(time.time() * 1000))

    compute_force(sess,ML_Force,atom_feature,atoms,norm_const)
    atoms.force /= md_const.f_scale

    t4 = int(round(time.time() * 1000))
    print('initial time to build nlist, feature and compute force: ',t2-t1,t3-t2,t4-t3)

    print("Step   dE/dT   Kinetic Energy   Temperature   Pressure Time")

    atoms.writexyz(0+md_param.icontinue)

    exit(1)
    for step_num in range(md_param.istep):
        t1 = int(round(time.time() * 1000))
        time_verlet(step_num,atoms,sess,ML_Force,md_const,md_param,norm_const)
        t2 = int(round(time.time() * 1000))
        if step_num % 50 == 0:
            atoms.writexyz(step_num+md_param.icontinue)
        if step_num % md_param.iprint == 0:
            cal_properties_ML(atoms,md_const)
            print("%10d \t  %10.6f \t  %10.6f \t %10.6f  \t %10.6f \t %10.6f \t %10.6f" %
                    (step_num+md_param.icontinue, md_const.DT*(step_num+md_param.icontinue),atoms.epot,atoms.ekin,
                     atoms.temperature,atoms.pressure,t2-t1))
            out_stat.write("%10d \t  %10.6f \t  %10.6f \t %10.6f  \t %10.6f \t %10.6f \t %10.6f \n"  %
                    (step_num+md_param.icontinue, md_const.DT*(step_num+md_param.icontinue),atoms.epot,atoms.ekin,
                     atoms.temperature, atoms.pressure, t2 - t1))
            force_vel_stat.write("%10d \t %10.6f \t  %10.6f \t %10.6f  \t %10.6f \t %10.6f \t %10.6f \t %10.6f \t %10.6f\n" %
                         (step_num+md_param.icontinue,atoms.tot_force[0],atoms.tot_force[1],atoms.tot_force[2],
                          atoms.tot_vel[0],atoms.tot_vel[1],atoms.tot_vel[2],atoms.temperature,atoms.epot))
    pool.close()
    pool.join()
    force_vel_stat.close()
    out_stat.close()

    md_param.icontinue += md_param.istep
    write_restart(atoms,md_const,md_param.icontinue)

if __name__ == '__main__':
    main()

