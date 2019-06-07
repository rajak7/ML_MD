import numpy as np

class constants:

    class ConstError(TypeError):
        pass

    def __init__(self):
        # units are defined in lammps metal unit
        self.BK = 8.617343e-5                      # Boltzmann constant (eV/K)
        self.PRESS_GPA = 60.21766208               # 1ev/A^3 ---> GPa
        self.KE_EV = 1.0364269e-4                  # mv2(gram/mole*(A/ps)^2) to energy(eV)
        self.EV_KE = 1.60218e-20                   # energy(eV)  --> mv2(gram*(A/ps)^2)
        self.FTM_VEL = 9648.533823273016           # convert (force/mass)*time ((eV/A)/(gram-mole))*ps--> velocity unit (A/ps)
        self.MASS_VOL_DEN = 1.6605389210321897     # conversion for mass/volume (gram/mole)/(A^3) ------> density  (gram/cm^3)
        self.ONE_MOLE = 6.023e23                   # number of atoms in 1 mole
        self.INV_ONE_MOLE = 1.0 / self.ONE_MOLE    # inverse of one mole
        self.KE_TEMP = 1.0 / (1.5 * self.BK)       # conversion factor to convert 0.5mv^2 to Temperature
        self.PI = 3.1415926

    def __setattr__(self,name,value):
        if name  in self.__dict__:
            print("%s is already defined as constant "% (name))
            raise self.ConstError
        else:
            self.__dict__[name] = value

    def __delattr__(self, name):
        if name in self.__dict__:
            print("deleting const parameter %s is not allowed " % (name))
            raise self.ConstError


#simulation specific parameters
class MD_Constants(constants):
    def __init__(self,f_scale,Mass,CUTOFF=8.0,DT=0.001):
        super().__init__()
        self.NCOMPONENT = len(Mass)
        self.MASS = np.asarray(Mass)
        self.ATOM_MASS =self.MASS * self.INV_ONE_MOLE
        self.ACCEL_FAC = self.FTM_VEL / self.MASS
        self.CUTOFF = CUTOFF             # in A
        self.CUTOFFSQ =  CUTOFF*CUTOFF   # in A^2
        self.DT = DT                     # ps
        self.f_scale = f_scale
        self.DT_param()

    def DT_param(self):
        self.DTSQ = self.DT
        self.DTI = 1.0/self.DT
        self.DTHALF = 0.50*self.DT


#Input simulation parameters for the MD simulation
class Simulation_Parameter():
    def __init__(self):
        self.ncell = np.empty([3],dtype='int')           # unit cell in each dimension
        self.boxmd = np.empty([3], dtype='float')        # boxmd in each dimension
        self.halfboxmd = np.empty([3], dtype='float')    # halfboxmd in each dimension
        self.istep = 1000                                # Total number of MD simulation to run
        self.iprint = 50
        self.ini_new = 1
        self.iscale_v = 1
        self.coeff_quench = 0.9
        self.iscale_step = 100
        self.req_temp = 10
        self.lattice_constant = 5.26
        self.icontinue = 0

#--------Parametets related to neighborlist creation ---------
class N_list_parameters:
    def __init__(self,NMAX=3072,LMAX=20,MAX_NEIGH=300):
        self.NMAX = NMAX              # NMAX: max number of atoms
        self.LMAX = LMAX              # max number of linked list (md_param.mc max value)
        self.MAX_NEIGH = MAX_NEIGH    # Maximum number of neighbors an atom can have
        self.ndir = [-1, 0, 1]        # used while creating neighbor list
        self.nheader = np.empty((self.LMAX, self.LMAX, self.LMAX), dtype=int);
        self.lsize = np.empty((self.LMAX, self.LMAX, self.LMAX), dtype=int)
        self.linklist = np.empty((self.NMAX), dtype=int)
        self.cellsize = np.empty([3], dtype='float')     # cellsize are the size of a reduced cell
        self.mc = np.empty([3], dtype='int')             # mc are the numbers of reduced cells

class MDatoms_properties(N_list_parameters):
    def __init__(self):
        super().__init__()
        self.boxmd = None
        self.localN = None
        self.pos = None
        self.atype = None
        self.vel = None
        self.force = None
        self.properties()

    def properties(self):
        self.volume = 0.0
        self.density = 0.0
        self.ekin = 0.0
        self.epot = 0.0
        self.etot = 0.0
        self.temperature = 0.0
        self.pressure = 0.0
        self.tot_force = [0.0,0.0,0.0]
        self.tot_vel = [0.0,0.0,0.0]

    def init_neighborlist(self):
        self.Neighbor = np.zeros((self.localN, self.MAX_NEIGH), dtype=int)


    def writexyz(self,step):
        with open('viz/output_' + str(step) + '.xyz', 'w') as outfile:
            outfile.write(str(self.localN) + '\n')
            outfile.write("%10.5f \t %10.5f \t %10.5f \n" %
                    (self.boxmd[0], self.boxmd[1], self.boxmd[2]))
            for val in range(self.localN):
                outfile.write("%4d \t %10.5f \t %10.5f \t %10.5f \t %10.5f \t %10.5f \t %10.5f\n" %
                    (self.atype[val],self.pos[val, 0], self.pos[val, 1], self.pos[val, 2],
                        self.force[val,0],self.force[val,1],self.force[val,2]))

    def initilize_sys(self,atype,pos,vel,boxmd,localN):
        self.atype = atype
        self.pos = pos
        self.vel = vel
        self.force = np.empty((localN, 3), dtype=float)
        self.localN = localN
        self.boxmd = boxmd
        self.properties()
        if self.localN > self.NMAX:
            print('localN is greater then NMAX: ',self.NMAX)
            exit(1)

    def compute_linkedlist_param(self,md_const):
        for val in range(3):
            self.mc[val] = int(self.boxmd[val] / md_const.CUTOFF)
            self.cellsize[val] = self.boxmd[val] / float(self.mc[val])
            if self.mc[val] > self.LMAX:
                print("linked list size is greater then LMAX: ",self.LMAX)
                exit(1)
        self.init_neighborlist()

    def cal_density(self,md_const):
        self.volume = self.boxmd[0] * self.boxmd[1] * self.boxmd[2]
        self.density = 0.0
        for i in range(self.localN):
            self.density += md_const.MASS[self.atype[i]]
        self.density = md_const.MASS_VOL_DEN * (self.density/self.volume)





