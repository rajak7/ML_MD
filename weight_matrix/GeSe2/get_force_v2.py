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
nfeatures = 310  # Number of radial & angular features
mean_x = mean[0:nfeatures] # mean & stddev only for x-direction
stddev_x = stddev[0:nfeatures]

xx = np.load('./train_XX.npy')
iatom=0
x1=xx[iatom][0:nfeatures]

print('xx&x1&mean&stddev: ', xx.shape,x1.shape,mean_x.shape, stddev_x.shape)

scaling_factor = 25.0
# correct the feature values by mean & stddev
xnew = (x1 - mean_x)/stddev_x
xnew *= scaling_factor

for layer in range(nlayers):
    #xnew = xnew.dot(w_x[layer])+b_x[layer]

    w = w_x[layer]
    b = b_x[layer]
    xtmp = np.zeros(b.shape[0])
    #print('shapes of w,b,xtmp,xnew: ', w.shape,b.shape,xtmp.shape,xnew.shape)
    for i in range(w.shape[1]):
        for j in range(w.shape[0]):
            xtmp[i] += xnew[j]*w[j][i]
        xtmp[i] += b[i]
    xnew = xtmp
    
    # relu except the last step
    if layer < nlayers - 1:
        xnew = np.maximum(xnew,0)
        print('relu on ', layer, '/',nlayers,' layer')

    print('layer,xnew.shape: ', layer, xnew.shape, '\t predicted: ', xnew[0], 
          '\tw&b shapes: ', w_x[layer].shape, b_x[layer].shape)

print('\npredicted force: ', xnew/scaling_factor) 
