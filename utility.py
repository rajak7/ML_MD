import numpy as np
import math
import random
import pickle

# create Monoatomic FCC Lattice
def createFCClattice(ncell,lat_constant):

    localN = 0
    currentr = [0.0]*3
    v1 = [0.5, 0.5, 0.0]
    v2 = [0.0, 0.5, 0.5]
    v3 = [0.5, 0.0, 0.5]
    Natoms = ncell[0]*ncell[1]*ncell[2]*4
    pos = np.empty((Natoms,3),dtype=float)
    atype = np.empty((Natoms), dtype=int)

    for i in range(ncell[2]):
        currentr[2] = float(i) * lat_constant + 1e-12
        for j in range(ncell[1]):
            currentr[1] = float(j) * lat_constant + 1e-12
            for k in range(ncell[0]):
                currentr[0] = float(k) * lat_constant + 1e-12
                for l in range(3):
                    pos[localN][l] = currentr[l]
                    pos[localN + 1][l] = currentr[l] + v1[l] * lat_constant
                    pos[localN + 2][l] = currentr[l] + v2[l] * lat_constant
                    pos[localN + 3][l] = currentr[l] + v3[l] * lat_constant
                    atype[localN] = 0
                    atype[localN + 1] = 0
                    atype[localN + 2] = 0
                    atype[localN + 3] = 0
                localN += 4
    if localN != Natoms:
        print("Inconsistant atom creation in createFCClattice ")
        exit(1)
    return atype,pos,localN

def readxyz_input(fname,boxmd):
    infile = open(fname,'r')
    localN = int(infile.readline().strip().split()[0])
    val = infile.readline().strip().split()
    boxmd[0] = float(val[0])
    boxmd[1] = float(val[1])
    boxmd[2] = float(val[2])

    pos = np.empty((localN, 3), dtype=float)
    atype = np.empty((localN), dtype=int)
    for count,val in enumerate(infile):
        val = val.strip().split()
        atype[count]  = int(val[0])
        pos[count][0] = float(val[1])
        pos[count][1] = float(val[2])
        pos[count][2] = float(val[3])
    return atype,pos,localN

# setup random velocities of atom at a desired temperature using Maxwell Distribution
def initilize_velocity(localN,atype,md_const,md_param):

    twopi = 2.0*md_const.PI
    kb_T = md_const.BK * md_param.req_temp * md_const.EV_KE

    #prepare Maxwell velocities
    facv = [0.0]*md_const.NCOMPONENT
    for i in range(md_const.NCOMPONENT):
        facv[i] = math.sqrt(2.0 * kb_T * md_const.ONE_MOLE / md_const.MASS[i])

    vel = np.empty((localN, 3), dtype=float)
    for i in range(localN):
        for j in range(3):
            rand_1 = random.random()
            rand_2 = random.random()
            vel[i][j] = facv[atype[i]] * math.sqrt(-math.log(rand_1)) * math.cos(twopi*rand_2)

    # make total momentum zero
    vel_avg = cal_linearmomentum(localN,vel,atype,md_const)

    for i in range(localN):
        for j in range(3):
            vel[i][j] -= vel_avg[j]

    #vel_avg = cal_linearmomentum(localN,vel,atype,md_const)
    #print(vel_avg)

    return vel

def cal_linearmomentum(localN,vel,atype,md_const):
    vcom = np.zeros((3),dtype='float')
    mtot = 0.0
    for i in range(localN):
        vcom[:] += md_const.MASS[atype[i]]*vel[i,:]
        mtot += md_const.MASS[atype[i]]
    return vcom/mtot


def scale_temp(md_param,md_const,atoms,itype=1):
    if itype == 1:
        cal_temperature(atoms,md_const)
        scale_t = math.sqrt(md_param.req_temp/atoms.temperature)
        print("Temperature scaled to: ",md_param.req_temp, "from ",atoms.temperature)
    elif itype == 2:
        scale_t = md_param.coeff_quench
        print("Temperature scaled by: ", scale_t, "from ",atoms.temperature)

    for i in range(atoms.localN):
        for j in range(3):
            atoms.vel[i,j] *= scale_t

    vel_avg = cal_linearmomentum(atoms.localN,atoms.vel,atoms.atype,md_const)
    for i in range(atoms.localN):
        for j in range(3):
            atoms.vel[i,j] -= vel_avg[j]

    cal_temperature(atoms,md_const)
    print("New Temperature: ", atoms.temperature)

def cal_temperature(atoms,md_const):
    atoms.ekin = 0.0
    for i in range(atoms.localN):
        ek_temp = atoms.vel[i,0]**2 + atoms.vel[i,1]**2 + atoms.vel[i,0]**2
        atoms.ekin += 0.5 * md_const.MASS[atoms.atype[i]] * ek_temp
    atoms.ekin = (atoms.ekin * md_const.KE_EV)/float(atoms.localN)
    atoms.temperature = atoms.ekin*md_const.KE_TEMP

def write_restart(atoms,md_const,step):
    output_file = open('MD.restart', 'wb')
    pickle.dump(atoms.localN, output_file)
    pickle.dump(atoms.boxmd, output_file)
    pickle.dump(step, output_file)
    pickle.dump(atoms.pos, output_file)
    pickle.dump(atoms.vel, output_file)
    pickle.dump(atoms.atype, output_file)
    output_file.close()

def read_restart(fname):
    input_file = open(fname,'rb')
    localN = pickle.load(input_file)
    boxmd = pickle.load(input_file)
    icontinue = pickle.load(input_file)
    pos = pickle.load(input_file)
    vel = pickle.load(input_file)
    atype = pickle.load(input_file)
    input_file.close()
    return atype,pos,vel,localN,boxmd,icontinue

def cal_properties_ML(atoms,md_const):
    #compute KE,Temperature and delta_PE
    atoms.epot = 0.0
    atoms.tot_force = [0.0,0.0,0.0]
    atoms.tot_vel = [0.0,0.0,0.0]
    atoms.pressure = 0.0
    cal_temperature(atoms, md_const)
    for i in range(atoms.localN):
        atoms.epot += atoms.force[i, 0]*atoms.vel[i, 0] + \
                      atoms.force[i, 1]*atoms.vel[i, 1] + atoms.force[i, 2]*atoms.vel[i, 2]
        atoms.pressure += atoms.force[i, 0]*atoms.pos[i, 0] + \
                          atoms.force[i, 1]*atoms.pos[i, 1] + atoms.force[i, 2]*atoms.pos[i, 2]
        for j in range(3):
            atoms.tot_force[j] += atoms.force[i,j]
    atoms.tot_vel = cal_linearmomentum(atoms.localN,atoms.vel,atoms.atype,md_const)
    atoms.pressure = 0.3333 * atoms.pressure + (atoms.localN * md_const.BK * atoms.temperature)
    atoms.pressure = (atoms.pressure / atoms.volume) * md_const.PRESS_GPA
    atoms.epot = -1.0 * atoms.epot * md_const.DT
    return
