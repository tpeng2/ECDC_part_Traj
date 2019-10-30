#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 10:50:05 2019

@author: tpeng2
"""
import sys
sys.path.append("/home/tpeng2/Dropbox/EVAPDROP/Models/Stochastic_Poly/func/")

from matplotlib.colors import LogNorm
import numpy as np
from scipy import sparse
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from pylab import *
from scipy.stats import gaussian_kde
from scipy.io import loadmat
from scipy.io import loadmat
#import functions
from func_calcTeq import calcTeq,qTtoRH
import numpy as np

#%% load data 
#data_path='../Stochastic_Poly/data/'
data_path='/scratch365/tpeng2/data/'
casename='r050ph10'
lors='long'
zp_in=np.loadtxt(open(data_path+lors+'_zp_in_'+casename+'.csv'),delimiter=",")
Tp_in=np.loadtxt(open(data_path+lors+'_Tp_in_'+casename+'.csv'),delimiter=",")
Tp_est=np.loadtxt(open(data_path+lors+'_Tp_est_'+casename+'.csv'),delimiter=",")
Tf_in=np.loadtxt(open(data_path+lors+'_Tf_in_'+casename+'.csv'),delimiter=",")
Tf_est=np.loadtxt(open(data_path+lors+'_Tf_est_'+casename+'.csv'),delimiter=",")
qf_in=np.loadtxt(open(data_path+lors+'_qf_in_'+casename+'.csv'),delimiter=",")
qf_est=np.loadtxt(open(data_path+lors+'_qf_est_'+casename+'.csv'),delimiter=",")
#% separating test and train
traj_entries=zp_in.shape[0]
t_len=zp_in.shape[1]
traj_ind=np.arange(traj_entries)
test_len=int(0.2*traj_entries)
train_len=traj_entries-test_len
test_ind=np.random.choice(traj_entries,test_len,replace=False)
train_ind = np.setdiff1d(traj_ind,test_ind)
#%% 
from keras import layers
from keras import models
from keras import optimizers

#
model = models.Sequential()


# ENCODER
#model.add(layers.Conv1D(240, 8, activation='relu',padding='valid', input_shape=(t_len,4)))
#model.add(layers.MaxPooling1D(5))
#model.add(layers.Conv1D(120, 12, activation='elu',padding='valid'))
#model.add(layers.MaxPooling1D(2))
##model.add(layers.Flatten())
##model.add(layers.Dense(32,activation='relu'))
#
## ENCODER
#model.add(layers.Conv1D(120, 12, activation='relu',padding='valid'))
#model.add(layers.UpSampling1D(2))
#model.add(layers.Conv1D(240, 8, activation='relu',padding='valid'))
#model.add(layers.UpSampling1D(5))
#model.add(layers.Flatten())
#model.add(layers.Dense(t_len,activation='relu'))
#model.add(layers.Reshape((t_len,4)))
#
#
#
#Adam=optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#model.compile(optimizer=Adam, loss='cosine', metrics=['accuracy'])
#model.summary()


model.add(layers.Conv1D(240, 8, activation='relu',padding='same', input_shape=(t_len,4)))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(120, 12, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(60, 16, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(48,20, activation='relu',padding='same'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(12,32, activation='relu',padding='same'))
model.add(layers.UpSampling1D(2))

model.add(layers.Conv1D(12, 32, activation='relu',padding='same'))
model.add(layers.UpSampling1D(2))
model.add(layers.Conv1D(48, 20, activation='relu',padding='same'))
model.add(layers.UpSampling1D(5))
model.add(layers.Conv1D(60, 16, activation='relu',padding='same'))
model.add(layers.UpSampling1D(5))
model.add(layers.Conv1D(120, 12, activation='relu',padding='same'))
model.add(layers.UpSampling1D(1))
model.add(layers.Conv1D(4, 8, activation='relu',padding='same'))
#model.add(layers.Flatten())
#model.add(layers.Dense(t_len*4,activation='relu'))
model.add(layers.Reshape((t_len,4)))
Adam=optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer=Adam, loss='cosine', metrics=['accuracy'])
model.summary()



#% reshape and stack data
# train
zp_scl=0.04;
Tf_scl=301.15;
qf_scl=0.0248;
Tp_scl=301.7;
input_train=np.zeros([train_len,t_len,4])
input_train[:,:,0]=zp_in[train_ind]/zp_scl
input_train[:,:,1]=Tf_in[train_ind]/Tf_scl
input_train[:,:,2]=qf_in[train_ind]/qf_scl
input_train[:,:,3]=Tp_in[train_ind]/Tp_scl
#input_train=input_train.reshape(train_len,1,t_len,4)

est_train=np.zeros([train_len,t_len,4])
est_train[:,:,0]=zp_in[train_ind]/zp_scl
est_train[:,:,1]=Tf_est[train_ind]/Tf_scl
est_train[:,:,2]=qf_est[train_ind]/qf_scl
est_train[:,:,3]=Tp_est[train_ind]/Tp_scl
#est_train=est_train.reshape(train_len,1,t_len,4)

#cor_train[:,:,1]=(input_train[:,:,1]-est_train[:,:,1])/np.mean((input_train[:,:,1]-est_train[:,:,1]))
#zero_train=np.zeros(cor_train.shape)
# test
input_test=np.zeros([test_len,t_len,4])
input_test[:,:,0]=zp_in[test_ind]/zp_scl
input_test[:,:,1]=Tf_in[test_ind]/Tf_scl
input_test[:,:,2]=qf_in[test_ind]/qf_scl
input_test[:,:,3]=Tp_in[test_ind]/Tp_scl
#input_test=input_test.reshape(test_len,1,t_len,4)


est_test=np.zeros([test_len,t_len,4])
est_test[:,:,0]=zp_in[test_ind]/zp_scl
est_test[:,:,1]=Tf_est[test_ind]/Tf_scl
est_test[:,:,2]=qf_est[test_ind]/qf_scl
est_test[:,:,3]=Tp_est[test_ind]/Tp_scl
#est_test=est_test.reshape(test_len,1,t_len,4)
cor_test=(input_test-est_test)
zero_test=np.zeros(cor_test.shape)

#%% make it to tensor type
input_train=input_train.astype('float32')
input_test=input_test.astype('float32')


model.fit(est_train[0:33,:,:],input_train[0:33,:,:],epochs=10,batch_size=train_len//2,verbose=0, initial_epoch=5)
test_mse_score, test_mae_score = model.evaluate(cor_test,cor_test)

print('test_mae_score=',test_mae_score)
#%% predict
Xnew=est_test[0:5,:,:]
Ynew = model.predict(Xnew)

#%%
#Xnew[:,:,0]=Xnew[:,:,0]*zp_scl
#Xnew[:,:,1]=Xnew[:,:,1]*Tf_scl
#Xnew[:,:,2]=Xnew[:,:,2]*qf_scl
#Xnew[:,:,3]=Xnew[:,:,3]*Tp_scl
#Ynew[:,:,0]=Ynew[:,:,0]*zp_scl
#Ynew[:,:,1]=Ynew[:,:,1]*Tf_scl
#Ynew[:,:,2]=Ynew[:,:,2]*qf_scl
#Ynew[:,:,3]=Ynew[:,:,3]*Tp_scl


plot(Xnew[1,0:450,0])
plot(Ynew[1,0:450,0])
