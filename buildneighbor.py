import numpy as np

def buildList(localN,NMAX,pos,nheader,lsize,linklist,mc,cellsize):
    lcoor = [0] * 3

    for i in range(mc[0]):
        for j in range(mc[1]):
            for k in range(mc[2]):
                nheader[i, j, k] = -1
                lsize[i, j, k] = 0

    for i in range(NMAX):
        linklist[i] = -1

    for i in range(localN):
        for j in range(3):
            lcoor[j] = int(pos[i][j] / cellsize[j])
            if lcoor[j] >= mc[j]:
                print("for linked list", j, " exceed boundary ", lcoor[j], " max value", mc[j], "position: ",pos[i][j])
                exit(1)
        linklist[i] = nheader[lcoor[0], lcoor[1], lcoor[2]]
        nheader[lcoor[0], lcoor[1], lcoor[2]] = i
        lsize[lcoor[0], lcoor[1], lcoor[2]] += 1
    return

def neighborlist(atoms,boxmd,halfboxmd,md_const):
    ig = [0, 0, 0]
    ic1 = [0, 0, 0]
    dr = np.zeros(3, dtype='float32')
    atoms.Neighbor[:, 0] = 0

    for iatom in range(atoms.localN):
        for j in range(3):
            ig[j] = int(atoms.pos[iatom][j]/atoms.cellsize[j])
        for ix in atoms.ndir:
            for iy in atoms.ndir:
                for iz in atoms.ndir:
                    ic1[0] = (ig[0] + ix + atoms.mc[0]) % atoms.mc[0]
                    ic1[1] = (ig[1] + iy + atoms.mc[1]) % atoms.mc[1]
                    ic1[2] = (ig[2] + iz + atoms.mc[2]) % atoms.mc[2]
                    jatom = atoms.nheader[ic1[0]][ic1[1]][ic1[2]]
                    if ic1[0] < 0 or ic1[0] >=  atoms.mc[0]:
                        print("ic1[0] is out of bound")
                        exit(1)
                    if ic1[1] < 0 or ic1[1] >= atoms.mc[1]:
                        print("ic1[0] is out of bound")
                        exit(1)
                    if ic1[2] < 0 or ic1[2] >= atoms.mc[2]:
                        print("ic1[0] is out of bound")
                        exit(1)
                    while jatom >= 0:
                        if not jatom == iatom:
                            rsq = 0.0
                            for k in range(3):
                                dr[k] = atoms.pos[iatom][k] - atoms.pos[jatom][k]
                                if (dr[k] > halfboxmd[k]): dr[k] -= boxmd[k]
                                if (dr[k] < -1.0* halfboxmd[k]): dr[k] += boxmd[k]
                                rsq += dr[k]*dr[k]
                            #print("jatom: ", jatom, iatom,rsq,md_param.CUTOFFSQ)
                            if rsq <= md_const.CUTOFFSQ:
                                atoms.Neighbor[iatom][0] += 1
                                if atoms.Neighbor[iatom][0] >=  atoms.MAX_NEIGH:
                                    print("EXCEED MAX Neighbor list: ",atoms.MAX_NEIGH)
                                    exit(1)
                                atoms.Neighbor[iatom][atoms.Neighbor[iatom][0]] = jatom
                        jatom = atoms.linklist[jatom]
    return

def neighborlist_0(atoms,boxmd,halfboxmd,md_const):

    dx = [0.0, 0.0, 0.0]
    atoms.Neighbor[:,0] = 0

    for iatom in range(atoms.localN-1):
        for jatom in range(iatom+1,atoms.localN):
            rsq = 0.0
            for k in range(3):
                dx[k] = atoms.pos[iatom][k] - atoms.pos[jatom][k]
                if dx[k] > halfboxmd[k]:
                    dx[k] = dx[k] - boxmd[k]
                if dx[k] < -1.0 * halfboxmd[k]:
                    dx[k] = dx[k] + boxmd[k]
                rsq += dx[k]*dx[k]
            if rsq <= md_const.CUTOFFSQ:
                atoms.Neighbor[iatom][0] += 1
                atoms.Neighbor[jatom][0] += 1
                atoms.Neighbor[iatom][atoms.Neighbor[iatom][0]] = jatom
                atoms.Neighbor[jatom][atoms.Neighbor[jatom][0]] = iatom
    return



