import numpy as np 
import math


class dataset():
    def __init__(self,feature_vec,labels,coordinate,atype):
        self.feature_vec = feature_vec
        self.labels = labels
        self.coordinate = coordinate
        self.atype = atype
        self.elements = len(feature_vec)
        self.N_feature= feature_vec.shape[1]  
        self.mean = 0.0
        self.stdval = 0.0
        self.f_mean = 0.0
        self.f_std = 0.0
        self.cal_mean_Std()

    def print_info(self):
        print("Number of elements: ",self.elements)
        print("Number of features: ",self.N_feature)
    
    def cal_mean_Std(self):
        self.mean = self.feature_vec.mean(axis = 0)
        self.stdval = self.feature_vec.std(axis =0 )
        self.f_mean = self.labels.mean(axis = 0)
        self.f_std = self.labels.std(axis = 0)
    
    def normalize_data(self):
        print("Normaizing data: ",self.feature_vec.shape,self.mean.shape)
        self.n_feature_vec = (self.feature_vec - self.mean)/self.stdval
        self.n_labels = (self.labels - self.f_mean)/self.f_std
    
    def remove_redundent(self,tag=0,dim_keep=0):
        if tag == 0: 
            dim_keep = self.stdval[:] != 0.0
            self.mean = self.mean[dim_keep]
            self.stdval = self.stdval[dim_keep]
        self.feature_vec = self.feature_vec[:,dim_keep]
        self.n_feature_vec = self.n_feature_vec[:,dim_keep]
        self.N_feature = self.feature_vec.shape[1]  
        return dim_keep


#read training data
def read_input(file1,file2,file3,file4,itype,mask_val=True):
    train_XX=np.load(file1)
    train_output=np.load(file2)
    train_corr=np.load(file3)
    train_atype = np.load(file4)
    print(type(train_XX),type(train_output),type(train_corr),type(train_atype))
    if mask_val == True:
        mask = train_atype[:] == itype
        train_atype = train_atype[mask]
        train_XX = train_XX[mask]
        train_output = train_output[mask]
        train_corr = train_corr[mask]       
    train_data = dataset(train_XX,train_output,train_corr,train_atype)
    return train_data