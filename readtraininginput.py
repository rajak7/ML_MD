import numpy as np
MAX_NEIGH = 800     #max neighbors per atom

def read_training_data(filename):
    with open(filename,'r') as inputfile:
        Natoms=int(inputfile.readline().strip())
        line=inputfile.readline().strip().split()
        boxsize=np.array(line)
        boxsize=boxsize.astype('float32')
        atype=np.empty((Natoms),dtype='int')
        position=np.empty((Natoms,3),dtype='float32')
        force=np.empty((Natoms,3),dtype='float32')
        ii=0
        for val in inputfile:
            val=val.strip().split()
            if val[0] == 'Ge':
                atype[ii] = 0
            elif val[0] == 'Sb':
                atype[ii] = 1
            elif val[0] == 'Te':
                atype[ii] = 2
            else:
                print("Invalid atom type")
                exit(1)
            position[ii,:] = np.array(val[1:4]).astype('float32')
            force[ii,:] = np.array(val[4:7]).astype('float32')
            ii += 1
    return Natoms,boxsize,atype,position,force

def makelinkedlist(Natoms,boxsize,cellsize,positions):

    ng = (boxsize / cellsize).astype('int')
    cellsize = boxsize/ng
    print("Boxsize:",boxsize," Cellsize:",cellsize," Num of Bins:",ng)
    lshd = np.full(ng,-1,dtype=int)
    llst = np.empty(Natoms,dtype=int)
    for ii in range(0,Natoms):
        ig = (positions[ii][:]/cellsize).astype('int')
        for val in range(3):
            if positions[ii][val] == boxsize[val] and ig[val] == ng[val]:
                ig[val] -=1
        if np.min(ig) < 0 or ig[0] >= ng[0] or ig[1] >=ng[1] or ig[2] >= ng[2]:
            print("atoms out of bound",ig,ng)
            exit(1)
        llst[ii] = lshd[ig[0]][ig[1]][ig[2]]
        lshd[ig[0]][ig[1]][ig[2]] = ii
    return lshd,llst,ng

def makeneighbourlist(lshd,llst,positions,boxsize,Natom,cellsize,nlist,Rcutsq):

    ndir = [-1, 0, 1]  # neighbor directions
    ic1 = [0,0,0]
    dr=np.zeros(3,dtype='float32')
    halfbox = 0.5*boxsize
    Neighbor = np.zeros((Natom, MAX_NEIGH), dtype=int)

    for ii in range(0,Natom):
        iatom = ii
        ig = (positions[ii][:]/cellsize).astype('int')
        for ix in ndir:
            for iy in ndir:
                for iz in ndir:
                    ic1[0] = (ig[0] + ix + nlist[0]) % nlist[0]
                    ic1[1] = (ig[1] + iy + nlist[1]) % nlist[1]
                    ic1[2] = (ig[2] + iz + nlist[2]) % nlist[2]
                    jatom = lshd[ic1[0]][ic1[1]][ic1[2]]
                    while jatom >=0:
                        rsq = 0.0
                        if not jatom == iatom:
                            for kk in range(0,3):
                                dr[kk] = positions[iatom][kk] - positions[jatom][kk]
                                if (dr[kk] > halfbox[kk]): dr[kk] -= boxsize[kk]
                                if (dr[kk] < -halfbox[kk]): dr[kk] += boxsize[kk]
                                rsq += dr[kk] * dr[kk]
                            if rsq <= Rcutsq:
                                Neighbor[iatom][0] += 1
                                Neighbor[iatom][ Neighbor[iatom][0]] = jatom
                        jatom = llst[jatom]
    return Neighbor

def makeneighbourlist_0(positions,boxsize,Natom,Rcutsq):
    dx = [0.0, 0.0, 0.0]
    halfbox = 0.5 * boxsize

    Neighbor = np.zeros((Natom, MAX_NEIGH), dtype=int)

    for iatom in range(Natom-1):
        for jatom in range(iatom+1,Natom):
            rsq = 0.0
            for k in range(3):
                dx[k] = positions[iatom, k] - positions[jatom, k]
                if dx[k] > halfbox[k]: dx[k] = dx[k] - boxsize[k]
                if dx[k] < -1.0 * halfbox[k]: dx[k] = dx[k] + boxsize[k]
                rsq += dx[k]*dx[k]
            if rsq <= Rcutsq:
                Neighbor[iatom][0] += 1
                Neighbor[jatom][0] += 1
                Neighbor[iatom][ Neighbor[iatom][0]] = jatom
                Neighbor[jatom][Neighbor[jatom][0]] = iatom
    return Neighbor



def writexyz(Natoms,position,Neighbor):

    with open('output.xyz','w') as outputfile:
        outputfile.write(str(Natoms) + "\n")
        outputfile.write(str(Natoms) + "\n")
        for ii in range(0,Natoms):
            outputfile.write("Ni %12.6f %12.6f %12.6f  %6d\n" % (position[ii][0], position[ii][1], position[ii][2], Neighbor[ii][0]))












