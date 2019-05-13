import matplotlib.pyplot as plt
import numpy as np
import sys

fname=sys.argv[1]
title = sys.argv[2]
out_image = sys.argv[3]

dis=list()
Ge_Ge=list()
Ge_Se=list()
Se_Se=list()
Sq_tot=list()

with open(fname,'r') as infile:
    _=infile.readline()
    for val in infile:
        val=val.strip().split()
        dis.append(float(val[0]))
        Ge_Ge.append(float(val[1]))
        Ge_Se.append(float(val[2]))
        Se_Se.append(float(val[4]))
        Sq_tot.append(float(val[5]))
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(10,12))

fig.suptitle(title,fontweight='bold',fontsize=20,color='black')
pp=ax1.plot(dis,Ge_Ge,lw=3,color='red')
#ax1.set_xlabel('MD RUN',fontweight='bold',fontsize=24)
ax1.set_ylabel('s(q)',fontweight='bold',fontsize=24)
ax1.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax1.legend(pp,['Ge-Ge'],loc='lower right',fontsize=20,fancybox=None, shadow=None)
ax1.set_ylim(-0.5,1.5)
ax1.set_xticks(np.arange(0,6,0.5))

pp=ax2.plot(dis,Ge_Se,lw=3,color='blue')
#ax2.set_xlabel('MD RUN',fontweight='bold',fontsize=24)
ax2.set_ylabel('s(q)',fontweight='bold',fontsize=24)
ax2.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax2.legend(pp,['Ge-Se'],loc='lower right',fontsize=20)
ax2.set_ylim(-0.5,1.5)
ax2.set_xticks(np.arange(0,6,0.5))

pp=ax3.plot(dis,Se_Se,lw=3,color='green')
ax3.set_xlabel('Distance(A)',fontweight='bold',fontsize=24)
ax3.set_ylabel('s(q)',fontweight='bold',fontsize=24)
ax3.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax3.legend(pp,['Se-Se'],loc='lower right',fontsize=20)
ax3.set_ylim(-0.5,1.5)
ax3.set_xticks(np.arange(0,6,0.5))

pp=ax4.plot(dis,Sq_tot,lw=3,color='m')
ax4.set_xlabel('Distance(A)',fontweight='bold',fontsize=24)
ax4.set_ylabel('s(q)',fontweight='bold',fontsize=24)
ax4.tick_params(axis = 'both', which = 'major', labelsize = 20)
ax4.legend(pp,['Sq_GeSe2'],loc='lower right',fontsize=20)
ax4.set_ylim(-0.5,1.5)
ax4.set_xticks(np.arange(0,6,0.5))

plt.savefig(out_image)
plt.show()
