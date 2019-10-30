#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 11:23:06 2019

@author: tpeng2
"""
import numpy as np
#Ta=TfL_in
#RH=RHL_est
#r_init=rpinitL;
#r_now=rpL_in;
Ta=301.15
RH=0.95
r_init=25e-6
r_now=24.8e-6


def calcFR(rp,Tf,qf):
    rhoa=1.194;
    Volp = pi2*2.0/3.0*rp**3
    rhop = (m_s+Volp*rhow)/Volp
    TfC = Tf-273.15
    einf = 610.94*EXP(17.6257*TfC/(TfC+243.04))
    q_inf = Mw/Ru*einf/Tf/rhoa
    RHa = qf/q_inf
    Lv =(25.0 - 0.02274*26.0)*10.0**5
    Eff_C = 2.0*Mw*Gam/(Ru*rhow*rp*part%Tf)
    Eff_S = Ion*Os*(Mw/Ms)*(Sal/1000.0)*part%radius**3/(rp**3) 
    return FR

g=9.8;rhoa=1.194;rhow=1000;nuf=1.537e-5;
Cpa=1006.0;Cpp=4179.0;#J/kg-K
CpaCpp=Cpa/Cpp;
Pra=0.715; 
Sc=0.615; #nuf/0.000024992 %Brian 7/29/14  (@25degC: Dm =0.000025 = 25E-6m2/s)
Mw=0.018015; #kg/mol %Brian 7/29/14 
Ru=8.3144621; #m**3*Pa/K/mol %Brian 7/29/14
Ms=0.05844; #kg/mol
Sal=34.0;#psu
Nup=2.0;Shp=2.0;Gam=7.28e-02; #N/m
Ion=2.0;Os=1.093;
Dw=nuf/Sc;ka=nuf/Pra; #molecular diffusivity of water and heat
TaC=Ta-273.15; #celcius
Lv=(25.0-0.02274*TaC)*10**5;
Volp=4.0/3.0*np.pi*r_now**3;
m_s=Sal/1000.0*(4.0/3.0*np.pi*r_init**3)*rhow;
#m_s=np.array([m_s]*size(Volp,axis=1)).transpose()
rhop=(rhow*Volp+m_s)/Volp; #Salt mass
esat=610.94*np.exp(17.6257*(Ta-273.15)/((Ta-273.15)+243.04));
#coefficient alpha and beta
ca=Mw*Lv/(Ru*Ta);
cb=esat/Ta*(Lv*Mw*Dw)/(Ru*ka);
#begin iterations, assuming r_p doesn't change; expression of y
Teq=Ta-1;Tpis=Ta;itrcount=0;#initialization
while abs(Tpis-Teq)>1e-8:
    Tpis=Teq;
    y=2.0*Mw*Gam/(Ru*rhow*r_now*Tpis)-Ion*Os*m_s*Mw/Ms/(Volp*rhop-m_s);
    DT=Teq-Ta;Teq=Ta-DT; 
    AM1=(2*Ta+cb-273.15)/(Ta+cb-273.15); #quadratic solution
    COEA=cb/Ta**2*(ca**2/2-AM1*ca+1)*np.exp(y);COEB=1+cb/Ta*(ca-1)*np.exp(y);COEC=-cb*(RH-np.exp(y));
    DT1=(-COEB+np.sqrt(COEB**2-4*COEA*COEC))/(2*COEA);DT2=(-COEB-np.sqrt(COEB**2-4*COEA*COEC))/(2*COEA);
    Teq=DT1+Ta;itrcount=itrcount+1;
TeqC=Teq-273.15;
rpis=r_now;req=r_now+0.1e-6;
icnt=0
#initialize
Volp_eq=4.0/3.0*np.pi*r_now**3;
y_req=2.0*Mw*Gam/(Ru*rhow*req*Teq)-Ion*Os*m_s*Mw/Ms/(Volp_eq*rhop-m_s);
while abs(RH-1-y_req).any()>1e-4:
    rpis=req;
    Volp_eq=4.0/3.0*np.pi*r_now**3;
    y_req=2.0*Mw*Gam/(Ru*rhow*req*Teq)-Ion*Os*m_s*Mw/Ms/(Volp_eq*rhop-m_s);#salt mass does not change
    g_rk=(RH-1)-y_req;
    dgdr=2.0*Mw*Gam/(Ru*rhow*req**2*Teq)-(Ion*Os*m_s*(Mw/Ms))/(Volp_eq*rhop-m_s)*(4*np.pi*req**2*rhop)
    req=req-g_rk/dgdr;
    icnt=icnt+1;print(icnt)
print(TeqC,req)

