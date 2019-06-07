Network design

0 fc1/fc/w:0 (405, 512)
1 fc1/fc/b:0 (512,)
2 fc2/fc/w:0 (512, 1024)
3 fc2/fc/b:0 (1024,)
4 fc3/fc/w:0 (1024, 512)
5 fc3/fc/b:0 (512,)
6 fc4/fc/w:0 (512, 1)
7 fc4/fc/b:0 (1,)


Radial Feature

mu=[1.5,2.0,2.3,2.5,3.0,3.3,3.5,3.7,4.0,4.25,4.5,4.75,5.0,5.5,6.0,6.5,7.0,7.5,8.0]
eta=[0.5,1.0,2.0,3.0,8.0]
RCUTOFF_DEFAULT = 10.0
RCUTOFF_DAMP = 15.0

Angular Feature

A_LAMDA=[-1,1]
A_ZETA=[2.0,4.0,6.0]
A_ETA = [2.0,0.5,0.2,0.05]
A_MU = [2.0,2.5,3.0,3.5,4.0]
RC_ANGULAR = 4.0
RC_Ang_DAMP = 8.0

Offet for radial and angular feature

r_feat:  radial feature per dimension (x,y,z) = 285
tot_rad: total radial feature = 3*285
a_feat: angulat feature per diemnsion = 120
tot_a: toal angular feature = 3*120

Proper way to concatinate feature for a single atom having 1215 feature vector

x:  train_x_XX = np.concatenate((trainXX[0,0:r_feat],trainXX[0,(tot_rad+0):(tot_rad+a_feat)]),axis=0)
y:  train_y_XX = np.concatenate((trainXX[:,r_feat:2*r_feat],trainXX[:,(tot_rad+a_feat):(tot_rad+2*a_feat)]),axis=0)
z:  train_z_XX = np.concatenate((trainXX[:,2*r_feat:3*r_feat],trainXX[:,(tot_rad+2*a_feat):(tot_rad+3*a_feat)]),axis=0)
