import numpy as np
from parameters import *
from utility import *

def read_inputparameters(filename='input_param.txt'):
    md_param = Simulation_Parameter()
    with open(filename, 'r') as in_file:
        val = in_file.readline().strip().split()  # line 1
        md_param.istep = int(val[0])
        md_param.iprint = int(val[1])
        md_param.ini_new = int(val[2])
        val = in_file.readline().strip().split()  # line 2
        md_param.iscale_v = int(val[0])
        md_param.coeff_quench = float(val[1])
        md_param.iscale_step = int(val[2])
        val = in_file.readline().strip().split()  # line 3
        md_param.req_temp = float(val[0])
        md_param.lattice_constant = float(val[1])
        val = in_file.readline().strip().split()  # line 4
        md_param.ncell[0] = int(val[0])
        md_param.ncell[1] = int(val[1])
        md_param.ncell[2] = int(val[2])
        val = in_file.readline().strip().split()  # line 5
        DT = float(val[0])
        CUTOFF = float(val[1])
        f_scale = float(val[2])
        val = in_file.readline()                  # line 6
        val = in_file.readline().strip().split()  # line 7
        Mass = list()
        for items in val:
            Mass.append(float(items))
        md_const = MD_Constants(f_scale,Mass,CUTOFF,DT)
    return md_param,md_const

def initilize():
    # read input parameters from md_param file
    md_param, md_const = read_inputparameters()
    atoms = MDatoms_properties()

    # create entire system if init_new is 1 otherwise read from restart file
    if md_param.ini_new == 1:
        print("creating new FCC monoatomic lattice")
        md_param.boxmd = md_param.ncell * md_param.lattice_constant
        atype, pos, localN = createFCClattice(md_param.ncell, md_param.lattice_constant)
        vel = initilize_velocity(localN, atype, md_const, md_param)
    elif md_param.ini_new == 2:
        print("Reading xyz file: ")
        atype, pos, localN = readxyz_input('input.xyz',md_param.boxmd)
        vel = initilize_velocity(localN, atype, md_const, md_param)
    else:
        print("Reading Restart file: ")
        atype,pos,vel,localN,boxmd,icontinue = read_restart('MD.restart')
        md_param.icontinue = icontinue
        md_param.boxmd[0] = boxmd[0]
        md_param.boxmd[1] = boxmd[1]
        md_param.boxmd[2] = boxmd[2]

    md_param.halfboxmd = 0.5 * md_param.boxmd

    atoms.initilize_sys(atype,pos,vel,md_param.boxmd, localN)
    atoms.cal_density(md_const)
    atoms.compute_linkedlist_param(md_const)

    print("Total number of atoms: ",atoms.localN)
    print("Simulation Box size: ", atoms.boxmd[0], atoms.boxmd[1], atoms.boxmd[2])
    print(atoms.mc,atoms.cellsize)
    print("Number of components: ",md_const.NCOMPONENT)
    print("Density: ", atoms.density, "gm/cm^3", "and volume : ", atoms.volume, "A^3")

    # initilize velocity for newly created system:
    if md_param.ini_new == 1 or  md_param.ini_new == 2:
        scale_temp(md_param, md_const, atoms,1)
    return md_param,md_const,atoms
