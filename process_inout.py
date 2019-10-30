#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:46:38 2019
Process input and output
@author: tpeng2
"""
# %%
# include user-defined function path
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
from func_calcTeq import estTeq,qTtoRH
# %%read DNS
casename='r050ph10';
datapath='/scratch365/tpeng2/data/'
zpS_in=np.genfromtxt(datapath+'short_zp_in_'+casename+'.csv',delimiter=',')
TpS_in=np.genfromtxt(datapath+'short_Tp_in_'+casename+'.csv',delimiter=',')
rpS_in=np.genfromtxt(datapath+'short_rp_in_'+casename+'.csv',delimiter=',')
TfS_in=np.genfromtxt(datapath+'short_Tf_in_'+casename+'.csv',delimiter=',')
qfS_in=np.genfromtxt(datapath+'short_qf_in_'+casename+'.csv',delimiter=',')
zpL_in=np.genfromtxt(datapath+'long_zp_in_'+casename+'.csv',delimiter=',')
TpL_in=np.genfromtxt(datapath+'long_Tp_in_'+casename+'.csv',delimiter=',')
rpL_in=np.genfromtxt(datapath+'long_rp_in_'+casename+'.csv',delimiter=',')
TfL_in=np.genfromtxt(datapath+'long_Tf_in_'+casename+'.csv',delimiter=',')
qfL_in=np.genfromtxt(datapath+'long_qf_in_'+casename+'.csv',delimiter=',')
#%% load matfile for raw estimation
fieldmat=loadmat('/home/tpeng2/Dropbox/mat/NTLP/r050ph10/r050ph10_20_52s.mat');
zgrid=fieldmat['z']; # dim: 130x1
zugrid=fieldmat['zz']; # dim: 130x1, for everything with dim nz+2
uxym = fieldmat['uxym'];# dim: 130x1
txym = fieldmat['txym'][:,0];# dim: 130x1
qxym = fieldmat['txym'][:,1];# dim: 130x1
radmean=fieldmat['radmean']; # dim: 128x1 #stored in centers of zgrid
#%% Generate raw estimation data
# initialization ptraj (time, z, Tp, rp, qp, qf)
# dimension 0: traj entry, dimension 1: time series
TpS_est=np.zeros(TpS_in.shape);
TfS_est=np.zeros(TfS_in.shape);
rpS_est=np.zeros(rpS_in.shape);
qfS_est=np.zeros(qfS_in.shape);
TpL_est=np.zeros(TpL_in.shape);
TfL_est=np.zeros(TfL_in.shape);
rpL_est=np.zeros(rpL_in.shape);
qfL_est=np.zeros(qfL_in.shape);

#look up from mat profile data: txym, uxym, qxym
TfS_est=np.interp(zpS_in,zugrid[:,0],txym);
TfL_est=np.interp(zpL_in,zugrid[:,0],txym);
qfS_est=np.interp(zpS_in,zugrid[:,0],qxym);
qfL_est=np.interp(zpL_in,zugrid[:,0],qxym);
RHS_est=qTtoRH(qfS_est,TfS_est);
RHL_est=qTtoRH(qfL_est,TfL_est);
rpinitS=rpS_in[:,0];
rpinitL=rpL_in[:,0];
# particle assumed to reach equilibrium immediately
TpS_est=estTeq(TfS_est,RHS_est,rpinitS)
TpL_est=estTeq(TfL_est,RHL_est,rpinitL)



# estimate particle radius, use 
## %%
#save_cvs_path='/scratch365/tpeng2/data/'
#np.savetxt(save_cvs_path+"short_Tp_est_"+casename+".csv", TpS_est, delimiter=",")
#np.savetxt(save_cvs_path+"short_Tf_est_"+casename+".csv", TfS_est, delimiter=",")
#np.savetxt(save_cvs_path+"short_qf_est_"+casename+".csv", qfS_est, delimiter=",")
#np.savetxt(save_cvs_path+"long_Tp_est_"+casename+".csv", TpL_est, delimiter=",")
#np.savetxt(save_cvs_path+"long_Tf_est_"+casename+".csv", TfL_est, delimiter=",")
#np.savetxt(save_cvs_path+"long_qf_est_"+casename+".csv", qfL_est, delimiter=",")
