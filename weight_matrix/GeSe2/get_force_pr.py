import numpy as np
b_x = []
b_x.append(np.load('./Ge_x_fc1_fc_b:0.npy'))
b_x.append(np.load('./Ge_x_fc2_fc_b:0.npy'))
b_x.append(np.load('./Ge_x_fc3_fc_b:0.npy'))
b_x.append(np.load('./Ge_x_fc4_fc_b:0.npy'))

w_x=[]
w_x.append(np.load('./Ge_x_fc1_fc_w:0.npy'))
w_x.append(np.load('./Ge_x_fc2_fc_w:0.npy'))
w_x.append(np.load('./Ge_x_fc3_fc_w:0.npy'))
w_x.append(np.load('./Ge_x_fc4_fc_w:0.npy'))

mean = np.load('./feature_mean_Si.npy')
stddev = np.load('./feature_std_Si.npy')

nlayers = 4 # total layers
# for each atom our of 930=3*310 , first 3*190 are radial features and the remaining 360 values are angular feature
nfeatures = 3*310  # Total Number of radial & angular features
mean_x = mean[0:nfeatures] # mean & stddev only for x-direction
stddev_x = stddev[0:nfeatures]

xx = np.load('./train_XX.npy')
iatom=0
x1=xx[iatom][0:nfeatures]

print('xx&x1&mean&stddev: ', xx.shape,x1.shape,mean_x.shape, stddev_x.shape)

# correct the feature values by mean & stddev
x2 = (x1 - mean_x)/stddev_x

print("x2",x2.shape)
scale = 25.0   # scaling factor for the force
r_feat = 190  # radial feature dimension
a_feat= 120  # angular feature dimension
tot_rad = 3*r_feat

xnew = np.concatenate((x2[0:r_feat],x2[(tot_rad+0):(tot_rad+a_feat)]),axis=0)

print("xnew",xnew.shape)

for layer in range(nlayers):
    xnew = xnew.dot(w_x[layer])+b_x[layer]
    
    # relu except the last step
    if layer < 3:
        xnew = np.maximum(xnew,0)
    print('layer,xnew.shape: ', layer, xnew.shape, '\t predicted: ', xnew[0], 
          '\tw&b shapes: ', w_x[layer].shape, b_x[layer].shape)

print('\npredicted force: ', xnew/scale) # scaling factor, 25x?
