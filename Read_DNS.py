#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read DNS printout and save files

Created on Wed Mar 13 15:15:39 2019

@author: tpeng2
"""
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
from func_calcTeq import calcTeq,qTtoRH
#%% local function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

#%%
# path used in Leibniz
grepdata_path='/home/tpeng2/postproc/partgrep/results'
# path used in mac
#grepdata_path='/Users/tpeng2/postproc/partgrep/results'
groupname='NTLP';casename='r050ph10';
maxpind=12
print('group: ',groupname);print('case: ',casename);
#%% load matfile for raw estimation
fieldmat=loadmat('/home/tpeng2/Dropbox/mat/NTLP/r050ph10/r050ph10_20_52s.mat');
zgrid=fieldmat['z']; # dim: 130x1
zugrid=fieldmat['zz']; # dim: 130x1, for everything with dim nz+2
uxym = fieldmat['uxym'];# dim: 130x1
txym = fieldmat['txym'][:,0];# dim: 130x1
qxym = fieldmat['txym'][:,1];# dim: 130x1
radmean=fieldmat['radmean']; # dim: 128x1 #stored in centers of zgrid
#%%    Particle flux ==> corrected 3/6/2019
# loading files
pflxtmp=np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_pflx.dat')
N=len(pflxtmp);Nt=N/3;
#roll array
p0=pflxtmp[:,1];p1=np.roll(pflxtmp[:,1],1);p2=np.roll(pflxtmp[:,1],2);Pall=p1+p2+p0; #triple length
Pnew=Pall[::3];Pt=pflxtmp[::3,0];
#discard first 10000
Nburin=10000;
delT=Pt[-1]-Pt[Nburin-1];
Ptcrop=Pt[Nburin-1:];Pnewcrop=Pnew[Nburin-1:];
dt=np.mean(np.diff(Pt));
#mean Np in one time step
#np.trapz(y,x), opposite input order vs. Matlab
Fdt=1/delT*np.trapz(Pnewcrop,x=Ptcrop);
F=Fdt/dt; print('Particle generation rate (F) is ',F,'per second')
#%% looping through samples
for i in range(maxpind):
    tmppind=i+1;tmppindst=str(tmppind);
    xptmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_xp'+tmppindst+'.dat')
    vptmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_vp'+tmppindst+'.dat')
    uftmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_uf'+tmppindst+'.dat')
    tLtmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_tL'+tmppindst+'.dat')
    TpTftmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_TpTf'+tmppindst+'.dat')
    qpqftmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_qpqf'+tmppindst+'.dat')
    rprpitmp = np.loadtxt(grepdata_path+'/'+groupname+'_'+casename+'_rp'+tmppindst+'.dat')
    rmzeroindT=np.where(TpTftmp[:,2]==0);rmzeroTind=np.array(rmzeroindT);
    rmzeroindq=np.where(qpqftmp[:,2]==0);rmzeroqind=np.array(rmzeroindq);
    TpTftmp[rmzeroTind,2]=TpTftmp[rmzeroTind,1];
    qpqftmp[rmzeroTind,2]=qpqftmp[rmzeroTind,1];

    
    print('reading sample No.'+tmppindst+' finished')
    difftL1=np.diff(tLtmp[:,1]);
    dpid=np.where(difftL1<0); newpidtmp=np.asarray(dpid)+1; newpidtmp=newpidtmp.ravel();
    print('last five new pid',newpidtmp[-5:-1])
    if i==0:
        newpid=newpidtmp;
        xp=xptmp[newpid[0]:newpid[-1]];
        vp=vptmp[newpid[0]:newpid[-1]];
        uf=uftmp[newpid[0]:newpid[-1]];
        tL=tLtmp[newpid[0]:newpid[-1]];
        TpTf=TpTftmp[newpid[0]:newpid[-1]];
        qpqf=qpqftmp[newpid[0]:newpid[-1]];
        rprpi=rprpitmp[newpid[0]:newpid[-1]];
        newpart=newpid-newpid[0];
        print(newpart[-1])
        #index for new particle, current index-the begining of newpid
        #the first index of newpart is 0
    else:    #appending new samples
        newpid=newpidtmp;
        xp=np.concatenate((xp,xptmp[newpid[0]:newpid[-1]]));
        vp=np.concatenate((vp,vptmp[newpid[0]:newpid[-1]]));
        uf=np.concatenate((uf,xptmp[newpid[0]:newpid[-1]]));
        tL=np.concatenate((tL,tLtmp[newpid[0]:newpid[-1]]));
        TpTf=np.concatenate((TpTf,TpTftmp[newpid[0]:newpid[-1]]));
        qpqf=np.concatenate((qpqf,qpqftmp[newpid[0]:newpid[-1]]));
        rprpi=np.concatenate((rprpi,rprpitmp[newpid[0]:newpid[-1]]));
        #new particle locations
        newpart=np.concatenate((newpart,newpid-newpid[0]+newpart[-1]),axis=None)
        #newpid in this batch - first newpid in this batch + last newpart[-1]+1
    print('crop sample No.'+tmppindst+' finished')

# make the new particle index array has only unique elements
newpart=np.unique(newpart)
#%% Trim trajectories
    # two parts: short and long tL
    # create two separate matrix
    # record and then reduce extra space
numtraj=len(newpart)-1;
# initialize ptraj (time, z, Tp, rp, qp, qf)
maxtrajlenS=200
maxtrajlenL=3000
# set time limit as 10 seconds
maxtrajtimeS=0.75
maxtrajtimeL=10
# evenly gridded time trajectory
time_traj_intpS=np.arange(0,maxtrajtimeS,maxtrajtimeS/maxtrajlenS)
time_traj_intpL=np.arange(0,maxtrajtimeL,maxtrajtimeL/maxtrajlenL)
# initialization
ptrajS_in=np.zeros([numtraj,maxtrajlenS])
ptrajS_out=np.zeros([numtraj,maxtrajlenS])
timeS_inout=np.zeros([numtraj,maxtrajlenS])
zpS_in=np.zeros([numtraj,maxtrajlenS])
TpS_in=np.zeros([numtraj,maxtrajlenS])
rpS_in=np.zeros([numtraj,maxtrajlenS])
TfS_in=np.zeros([numtraj,maxtrajlenS])
qfS_in=np.zeros([numtraj,maxtrajlenS])
TpS_out=np.zeros([numtraj,maxtrajlenS])
rpS_out=np.zeros([numtraj,maxtrajlenS])
tLmaxS=np.zeros(numtraj);
nztLS=np.zeros(numtraj);
# longer traj
ptrajL_in=np.zeros([numtraj,maxtrajlenL])
ptrajL_out=np.zeros([numtraj,maxtrajlenL])
timeL_inout=np.zeros([numtraj,maxtrajlenL])
zpL_in=np.zeros([numtraj,maxtrajlenL])
TpL_in=np.zeros([numtraj,maxtrajlenL])
rpL_in=np.zeros([numtraj,maxtrajlenL])
TfL_in=np.zeros([numtraj,maxtrajlenL])
qfL_in=np.zeros([numtraj,maxtrajlenL])
TpL_out=np.zeros([numtraj,maxtrajlenL])
rpL_out=np.zeros([numtraj,maxtrajlenL])
tLmaxL=np.zeros(numtraj);
nztLL=np.zeros(numtraj);
tLcatind=np.zeros(numtraj);#1: large, 0 :small
#initialization finished, now assign numbers
for i in range(numtraj-1): # from 0 to 
    trajlen=newpart[i+1]-newpart[i]+1 #
    tLmaxS[i]=tL[newpart[i+1]-1,1];   #trajectory 
    tLmaxL[i]=tL[newpart[i+1]-1,1];   #trajectory 
    if tLmaxS[i]<=maxtrajtimeS:#linearly interpolate time
        tLcatind[i]=0;
        time_traj_orig=tL[newpart[i]:newpart[i+1],1]
        zp_in_orig=xp[newpart[i]:newpart[i+1],3]
        Tp_in_orig=TpTf[newpart[i]:newpart[i+1],1]
        rp_in_orig=rprpi[newpart[i]:newpart[i+1],1]
        Tf_in_orig=TpTf[newpart[i]:newpart[i+1],2]
        qf_in_orig=qpqf[newpart[i]:newpart[i+1],2]
        [time_nrst,idx_nrst]=find_nearest(time_traj_intpS,time_traj_orig[-1]);
        zpS_in[i,0:idx_nrst+1]=np.interp(time_traj_intpS[0:idx_nrst+1],time_traj_orig,np.copy(zp_in_orig));
        TpS_in[i,0:idx_nrst+1]=np.interp(time_traj_intpS[0:idx_nrst+1],time_traj_orig,np.copy(Tp_in_orig));
        rpS_in[i,0:idx_nrst+1]=np.interp(time_traj_intpS[0:idx_nrst+1],time_traj_orig,np.copy(rp_in_orig));
        TfS_in[i,0:idx_nrst+1]=np.interp(time_traj_intpS[0:idx_nrst+1],time_traj_orig,np.copy(Tf_in_orig));
        qfS_in[i,0:idx_nrst+1]=np.interp(time_traj_intpS[0:idx_nrst+1],time_traj_orig,np.copy(qf_in_orig));
#        print(i,TfS_in[i,0:idx_nrst+1])
    else: #linear interpolate# first, reconstruct the time
        tLcatind[i]=1;
        time_traj_orig=tL[newpart[i]:newpart[i+1],1]
        zp_in_orig=xp[newpart[i]:newpart[i+1],3]
        Tp_in_orig=TpTf[newpart[i]:newpart[i+1],1]
        rp_in_orig=rprpi[newpart[i]:newpart[i+1],1]
        Tf_in_orig=TpTf[newpart[i]:newpart[i+1],2]
        qf_in_orig=qpqf[newpart[i]:newpart[i+1],2]
        [time_nrst,idx_nrst]=find_nearest(time_traj_intpL,time_traj_orig[-1]);
        zpL_in[i,0:idx_nrst+1]=np.interp(time_traj_intpL[0:idx_nrst+1],time_traj_orig,zp_in_orig);
        TpL_in[i,0:idx_nrst+1]=np.interp(time_traj_intpL[0:idx_nrst+1],time_traj_orig,Tp_in_orig);
        rpL_in[i,0:idx_nrst+1]=np.interp(time_traj_intpL[0:idx_nrst+1],time_traj_orig,rp_in_orig);
        TfL_in[i,0:idx_nrst+1]=np.interp(time_traj_intpL[0:idx_nrst+1],time_traj_orig,Tf_in_orig);
        qfL_in[i,0:idx_nrst+1]=np.interp(time_traj_intpL[0:idx_nrst+1],time_traj_orig,qf_in_orig);
#        print(i,qfL_in)
    #nztL[i]=np.count_nonzero(zp_in[i,:])

# %%zero is annoying, so replace zero in temperature and humidity to the initial value
TpS_in[np.where(TpS_in==0)[0],np.where(TpS_in==0)[1]]=TpS_in[np.where(TpS_in==0)[0],0]
rpS_in[np.where(rpS_in==0)[0],np.where(rpS_in==0)[1]]=rpS_in[np.where(rpS_in==0)[0],0]
TfS_in[np.where(TfS_in==0)[0],np.where(TfS_in==0)[1]]=TfS_in[np.where(TfS_in==0)[0],0]
qfS_in[np.where(qfS_in==0)[0],np.where(qfS_in==0)[1]]=qfS_in[np.where(qfS_in==0)[0],0]
TpL_in[np.where(TpL_in==0)[0],np.where(TpL_in==0)[1]]=TpL_in[np.where(TpL_in==0)[0],0]
rpL_in[np.where(rpL_in==0)[0],np.where(rpL_in==0)[1]]=rpL_in[np.where(rpL_in==0)[0],0]
TfL_in[np.where(TfL_in==0)[0],np.where(TfL_in==0)[1]]=TfL_in[np.where(TfL_in==0)[0],0]
qfL_in[np.where(qfL_in==0)[0],np.where(qfL_in==0)[1]]=qfL_in[np.where(qfL_in==0)[0],0]

# %% 
indshort=np.where(tLcatind==0)[0];
indlong=np.where(tLcatind==1)[0];

#zpS_in[~np.all(zpL_in == 0, axis=1)]
zpS_in=zpS_in[~np.all(zpS_in == 0, axis=1)&(~np.all(qfS_in == 0, axis=1))]
TpS_in=TpS_in[~np.all(TpS_in == 0, axis=1)&(~np.all(qfS_in == 0, axis=1))]
rpS_in=rpS_in[~np.all(rpS_in == 0, axis=1)&(~np.all(qfS_in == 0, axis=1))]
TfS_in=TfS_in[(~np.all(TfS_in == 0, axis=1))&(~np.all(qfS_in == 0, axis=1))]
qfS_in=qfS_in[~np.all(qfS_in == 0, axis=1)]
zpL_in=zpL_in[~np.all(zpL_in == 0, axis=1)]
TpL_in=TpL_in[~np.all(TpL_in == 0, axis=1)]
rpL_in=rpL_in[~np.all(rpL_in == 0, axis=1)]
TfL_in=TfL_in[~np.all(TfL_in == 0, axis=1)]
qfL_in=qfL_in[~np.all(qfL_in == 0, axis=1)]

# %% save csv
#save_cvs_path='/home/tpeng2/scratch365/tpeng2/'
save_cvs_path='/scratch365/tpeng2/data/'
np.savetxt(save_cvs_path+"short_zp_in_"+casename+".csv", zpS_in, delimiter=",")
np.savetxt(save_cvs_path+"short_Tp_in_"+casename+".csv", TpS_in, delimiter=",")
np.savetxt(save_cvs_path+"short_rp_in_"+casename+".csv", rpS_in, delimiter=",")
np.savetxt(save_cvs_path+"short_Tf_in_"+casename+".csv", TfS_in, delimiter=",")
np.savetxt(save_cvs_path+"short_qf_in_"+casename+".csv", qfS_in, delimiter=",")
np.savetxt(save_cvs_path+"long_zp_in_"+casename+".csv", zpL_in, delimiter=",")
np.savetxt(save_cvs_path+"long_Tp_in_"+casename+".csv", TpL_in, delimiter=",")
np.savetxt(save_cvs_path+"long_rp_in_"+casename+".csv", rpL_in, delimiter=",")
np.savetxt(save_cvs_path+"long_Tf_in_"+casename+".csv", TfL_in, delimiter=",")
np.savetxt(save_cvs_path+"long_qf_in_"+casename+".csv", qfL_in, delimiter=",")
np.savetxt(save_cvs_path+"short_time_"+casename+".csv", time_traj_intpS, delimiter=",")
np.savetxt(save_cvs_path+"long_time_"+casename+".csv", time_traj_intpL, delimiter=",")

