#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:23:16 2019
Calc Teq
@author: tpeng2
"""
import numpy as np
# calculate equilibrium temperature and radius (1996)
def calcTeq(Ta,RH,r_init,r_now):
    global g,rhoa,rhow,nuf
    global Cpa,Cpp, CpaCpp
    global Pra,Sc,Mw,Ru,Ms,Sal,Nup,Shp,Gam,Ion,Os,Dw,ka
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
    if m_s.shape[0]>0:
        m_s=np.array([m_s]*np.size(Volp,axis=1)).transpose()
    rhop=(rhow*Volp+m_s)/Volp; #Salt mass
    esat=610.94*np.exp(17.6257*(Ta-273.15)/((Ta-273.15)+243.04));
    #coefficient alpha and beta
    ca=Mw*Lv/(Ru*Ta);
    cb=esat/Ta*(Lv*Mw*Dw)/(Ru*ka);
    #begin iterations, assuming r_p doesn't change; expression of y
    Teq=Ta-1;Tpis=Ta;itrcount=0;#initialization
    while abs(Tpis-Teq).any()>1e-8:
        Tpis=Teq;
        y=2.0*Mw*Gam/(Ru*rhow*r_now*Tpis)-Ion*Os*m_s*Mw/Ms/(Volp*rhop-m_s);
        DT=Teq-Ta;Teq=Ta-DT; 
        AM1=(2*Ta+cb-273.15)/(Ta+cb-273.15); #quadratic solution
        COEA=cb/Ta**2*(ca**2/2-AM1*ca+1)*np.exp(y);COEB=1+cb/Ta*(ca-1)*np.exp(y);COEC=-cb*(RH-np.exp(y));
        DT1=(-COEB+np.sqrt(COEB**2-4*COEA*COEC))/(2*COEA);DT2=(-COEB-np.sqrt(COEB**2-4*COEA*COEC))/(2*COEA);
        Teq=DT1+Ta;itrcount=itrcount+1;
    TeqC=Teq-273.15;
    return Teq
#%%
def estTeq(Ta,RH,r_init):
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
    Volp=4.0/3.0*np.pi*r_init**3;
    m_s=Sal/1000.0*(4.0/3.0*np.pi*r_init**3)*rhow;
    if m_s.shape[0]>0:
        m_s=np.array([m_s]*np.size(Ta,axis=1)).transpose()    
        Volp=np.array([Volp]*np.size(Ta,axis=1)).transpose()    
    rhop=(rhow*Volp+m_s)/Volp; #Salt mass
    esat=610.94*np.exp(17.6257*(Ta-273.15)/((Ta-273.15)+243.04));
    #coefficient alpha and beta
    ca=Mw*Lv/(Ru*Ta);
    cb=esat/Ta*(Lv*Mw*Dw)/(Ru*ka);
    #begin iterations, assuming r_p doesn't change; expression of y
    Teq=Ta-1;Tpis=Ta;itrcount=0;#initialization
    while abs(Tpis-Teq).any()>1e-8:
        Tpis=Teq;
        r_now=np.array([r_init]*np.size(Ta,axis=1)).transpose();
        y=2.0*Mw*Gam/(Ru*rhow*r_now*Tpis)-Ion*Os*m_s*Mw/Ms/(Volp*rhop-m_s);
        DT=Teq-Ta;Teq=Ta-DT; 
        AM1=(2*Ta+cb-273.15)/(Ta+cb-273.15); #quadratic solution
        COEA=cb/Ta**2*(ca**2/2-AM1*ca+1)*np.exp(y);COEB=1+cb/Ta*(ca-1)*np.exp(y);COEC=-cb*(RH-np.exp(y));
        DT1=(-COEB+np.sqrt(COEB**2-4*COEA*COEC))/(2*COEA);DT2=(-COEB-np.sqrt(COEB**2-4*COEA*COEC))/(2*COEA);
        Teq=DT1+Ta;itrcount=itrcount+1;
    TeqC=Teq-273.15;
    return Teq

#%%
def calcreq(Ta,RH,r_init):
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
    Volp=4.0/3.0*np.pi*r_init**3;
    m_s=Sal/1000.0*(4.0/3.0*np.pi*r_init**3)*rhow;
    #m_s=np.array([m_s]*np.size(Volp,axis=1)).transpose()
    rhop=(rhow*Volp+m_s)/Volp; #Salt mass
    esat=610.94*np.exp(17.6257*(Ta-273.15)/((Ta-273.15)+243.04));
    #coefficient alpha and beta
    ca=Mw*Lv/(Ru*Ta);
    cb=esat/Ta*(Lv*Mw*Dw)/(Ru*ka);
    #begin iterations, assuming r_p doesn't change; expression of y
    Teq=Ta-1;Tpis=Ta;itrcount=0;#initialization
    rpis=r_init;req=r_init-0.2e-6;
    while abs(rpis-req)>1e-8:
        rpis=req;
        Volp_eq=4.0/3.0*np.pi*req**3;
        print('Volp=',Volp)
        y_req=2.0*Mw*Gam/(Ru*rhow*req*Teq)-Ion*Os*m_s*Mw/Ms/(Volp_eq*rhop-m_s);#salt mass does not change
        g_rk=(RH-1)-y_req;
        dgdr=2.0*Mw*Gam/(Ru*rhow*req**2*Teq)-Ion*Os*m_s*Mw/Ms/(Volp_eq*rhop-m_s)*(4*np.pi*req**2*rhop)
        req=req-g_rk/dgdr;
    return req

# for Tf and qf
def qTtoRH(qf,Tf):
    Mw=0.018015; #kg/mol %Brian 7/29/14 
    Ru=8.3144621; #m**3*Pa/K/mol %Brian 7/29/14
    rhoa=1.1;
    TfC=Tf-273.15;
    qs = Mw/Ru/Tf*610.94*np.exp(17.6257*TfC/(TfC+243.04))/rhoa;
    RH=qf/qs;
    return RH

