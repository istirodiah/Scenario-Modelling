#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 12:24:13 2024

@author: istirodiah
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy.optimize import Bounds
bounds = Bounds(np.zeros((7)), 100*np.ones((7)))


def model(init_vals, params, opparams, t, rho):
    S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0 = init_vals
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]
    c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, r = params
    
    p        = np.zeros(7)
    beta     = np.zeros(7)
    p[3:]    = opparams[3:]
    beta[:3] = opparams[:3]
    
    beta1  = beta
    beta2  = beta * 0.75
    beta3  = beta * 0.5
    beta4  = beta * 0.25
    betav  = beta * 0.5
    rho1   = rho
    rho2   = rho * 0.75
    rho3   = rho * 0.5
    rho4   = rho * 0.25
    rho5   = rho * 0.1
    sigma1 = sigma
    sigma2 = sigma * 0.75
    sigma3 = sigma * 0.5
    sigma4 = sigma * 0.25
    sigma5 = sigma * 0.1
    phi1   = phi
    phi2   = phi * 0.75
    phi3   = phi * 0.5
    phi4   = phi * 0.25
    phi5   = phi * 0.1
    
    dt = (t[1] - t[0])*1./7
    
    for i in t[1:]:
        
        next_Ns  = Ns[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta1*sea*S1[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta2*sea*S2[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta3*sea*S3[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta4*sea*S4[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*betav*sea*V[-1] + p*S5[-1])*dt
        next_Nh  = Nh[-1] + (rho1*I1[-1] + rho2*I2[-1] + rho3*I3[-1] + rho4*I4[-1] + rho5*I5[-1])*dt
       
        next_S1 = S1[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta1*sea*S1[-1] + epsilon*S1[-1])*dt
        next_E1 = E1[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta1*sea*S1[-1] - alpha*E1[-1])*dt
        next_I1 = I1[-1] + (alpha*E1[-1] - (theta + rho1 + sigma1)*I1[-1])*dt
        next_H1 = H1[-1] + (rho1*I1[-1] - (eta + phi1)*H1[-1])*dt
        next_R1 = R1[-1] + (eta*H1[-1] + theta*I1[-1] - gamma*R1[-1])*dt
        
        next_S2 = S2[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta2*sea*S2[-1] - mu*V[-1] - gamma*R1[-1])*dt
        next_E2 = E2[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta2*sea*S2[-1] - alpha*E2[-1])*dt
        next_I2 = I2[-1] + (alpha*E2[-1] - (theta + rho2 + sigma2)*I2[-1])*dt
        next_H2 = H2[-1] + (rho2*I2[-1] - (eta + phi2)*H2[-1])*dt
        next_R2 = R2[-1] + (eta*H2[-1] + theta*I2[-1] - gamma*R2[-1])*dt
        
        next_S3 = S3[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta3*sea*S3[-1] - gammav*Rv[-1] - gamma*R2[-1])*dt
        next_E3 = E3[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta3*sea*S3[-1] - alpha*E3[-1])*dt
        next_I3 = I3[-1] + (alpha*E3[-1] - (theta + rho3 + sigma3)*I3[-1])*dt
        next_H3 = H3[-1] + (rho3*I3[-1] - (eta + phi3)*H3[-1])*dt
        next_R3 = R3[-1] + (eta*H3[-1] + theta*I3[-1] - gamma*R3[-1])*dt
        
        next_S4 = S4[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta4*sea*S4[-1] - gamma*R3[-1])*dt
        next_E4 = E4[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta4*sea*S4[-1] - alpha*E4[-1])*dt
        next_I4 = I4[-1] + (alpha*E4[-1] - (theta + rho4 + sigma4)*I4[-1])*dt
        next_H4 = H4[-1] + (rho4*I4[-1] - (eta + phi4)*H4[-1])*dt
        next_R4 = R4[-1] + (eta*H4[-1] + theta*I4[-1])*dt
        
        next_S5 = S5[-1] - (p*S5[-1] - gamma*R4[-1] - gamma*R5[-1])*dt
        next_E5 = E5[-1] + (p*S5[-1] - alpha*E5[-1])*dt
        next_I5 = I5[-1] + (alpha*E5[-1] - (theta + rho5 + sigma5)*I5[-1])*dt
        next_H5 = H5[-1] + (rho4*I5[-1] - (eta + phi5)*H5[-1])*dt
        next_R5 = R5[-1] + (eta*H5[-1] + theta*I5[-1] - gamma*R5[-1])*dt
        
        next_D  = D[-1] + (phi1*H1[-1] + sigma1*I1[-1] + phi2*H2[-1] + sigma2*I2[-1] + phi3*H3[-1] + sigma3*I3[-1]+ phi4*H4[-1] + sigma4*I4[-1] + phi5*H5[-1] + sigma5*I5[-1])*dt
        
        next_V  = V[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*betav*sea*V[-1] + mu*V[-1] - epsilon*S1[-1])*dt
        next_Ev = Ev[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*betav*sea*V[-1] - alpha*Ev[-1])*dt
        next_Iv = Iv[-1] + (alpha*Ev[-1] - theta*Iv[-1])*dt
        next_Rv = Rv[-1] + (theta*Iv[-1] - gammav*Rv[-1])*dt
        
        
        Ns = np.vstack((Ns, next_Ns))
        Nh = np.vstack((Nh, next_Nh))
        
        S1 = np.vstack((S1, next_S1))
        E1 = np.vstack((E1, next_E1))
        I1 = np.vstack((I1, next_I1))
        H1 = np.vstack((H1, next_H1))
        R1 = np.vstack((R1, next_R1))
        S2 = np.vstack((S2, next_S2))
        E2 = np.vstack((E2, next_E2))
        I2 = np.vstack((I2, next_I2))
        H2 = np.vstack((H2, next_H2))
        R2 = np.vstack((R2, next_R2))
        S3 = np.vstack((S3, next_S3))
        E3 = np.vstack((E3, next_E3))
        I3 = np.vstack((I3, next_I3))
        H3 = np.vstack((H3, next_H3))
        R3 = np.vstack((R3, next_R3))
        S4 = np.vstack((S4, next_S4))
        E4 = np.vstack((E4, next_E4))
        I4 = np.vstack((I4, next_I4))
        H4 = np.vstack((H4, next_H4))
        R4 = np.vstack((R4, next_R4))
        S5 = np.vstack((S5, next_S5))
        E5 = np.vstack((E5, next_E5))
        I5 = np.vstack((I5, next_I5))
        H5 = np.vstack((H5, next_H5))
        R5 = np.vstack((R5, next_R5))
        D  = np.vstack((D, next_D))
        V  = np.vstack((V, next_V))
        Ev = np.vstack((Ev, next_Ev))
        Iv = np.vstack((Iv, next_Iv))
        Rv = np.vstack((Rv, next_Rv))
    
    return S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh


def cost(opparams, modelparams, idata, i):
    init_vals, params, N, t = modelparams
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = model(init_vals, params, opparams, t)
        
    dum = 0
    
    dui  = sum(idata[:i+1])
    simi = Ns[7,:] * N
    if max(dui) == 0:
        p = (simi - dui)**2
    else:
        p = (simi - dui)**2 * 1./max(dui)
    
    dum = dum + sum(p) 
    return dum


file   = pd.ExcelFile('RSV.xlsx')

df1    = file.parse(30)
idata  = df1.values[0:, 2:9]
df2    = file.parse(18) 
hdata  = df2.values[0:, 1:8]

it = 104 
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters		

c   = np.array([[1.046E-07, 8.024E-08, 6.986E-08, 6.114E-08, 4.288E-08, 1.957E-08, 0.000E+00],          #covimod contact rate
                      [8.024E-08, 3.687E-07, 8.406E-08, 3.989E-08, 5.159E-08, 3.517E-08, 7.997E-08],
                      [6.986E-08, 8.406E-08, 5.721E-08, 2.569E-08, 2.176E-08, 1.171E-08, 2.647E-08],
                      [6.114E-08, 3.989E-08, 2.569E-08, 2.942E-08, 1.908E-08, 1.274E-08, 4.122E-08],
                      [4.288E-08, 5.159E-08, 2.176E-08, 1.908E-08, 1.418E-08, 8.350E-09, 1.447E-08],
                      [1.957E-08, 3.517E-08, 1.171E-08, 1.274E-08, 8.350E-09, 1.488E-08, 1.441E-08],
                      [0.000E+00, 7.997E-08, 2.647E-08, 4.122E-08, 1.447E-08, 1.441E-08, 2.858E-08]])


sea     = 1   # change below
rho     = np.array([0.25, 0.15, 0.08, 0.06, 0.05, 0.05, 0.05])
theta   = 1./10 
sigma   = 0
gamma   = 1./30
gammav  = 1./30
mu      = 1./90
alpha   = 1./7
eta     = 1./5
phi     = 0
epsilon = np.array([0.05, 0.05, 0, 0, 0, 0, 0])

params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho

N = 83166711         # worldometer data
N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.3709, 0.2057, 0.0428])

# Init (change to last running)
Ns_0 = idata[0]/N
Nh_0 = np.zeros(7)

E_0  = Ns_0 
I_0  = 2*Ns_0 
H_0  = Nh_0
R_0  = np.zeros(7)

Eb = np.zeros(7)
Eu = np.zeros(7)
Eb[:3] = E_0[:3] 
Eu[3:] = E_0[3:]

Ib = np.zeros(7)
Iu = np.zeros(7)
Ib[:3] = I_0[:3] 
Iu[3:] = I_0[3:]

Hb = np.zeros(7)
Hu  = np.zeros(7)
Hb[:3] = H_0[:3] 
Hu[3:] = H_0[3:]

E1_0 = Eb * 5./10 
I1_0 = Ib * 4./10
H1_0 = Hb * 0.6
R1_0 = np.zeros(7)
E2_0 = Eb * 3./10 
I2_0 = Ib * 4./10
H2_0 = Hb * 0.3
R2_0 = np.zeros(7)
E3_0 = Eb * 2./10 
I3_0 = Ib * 2./10 
H3_0 = Hb * 0.1
R3_0 = np.zeros(7)
E4_0 = Eb * 0 
I4_0 = Ib * 0 
H4_0 = Hb * 0
R4_0 = np.zeros(7)
E5_0 = Eu  
I5_0 = Iu 
H5_0 = Hu 
R5_0 = np.zeros(7)
D_0  = np.zeros(7)
V_0  = 1./N * np.array([0.009, 0, 0, 0, 0, 0, 0])
Ev_0 = np.zeros(7)
Iv_0 = np.zeros(7)
Rv_0 = np.zeros(7)

S_0  = N*1./N - (E_0+I_0+H_0+R_0+D_0+V_0+Ev_0+Iv_0+Rv_0)

Sb = np.zeros(7)
Su = np.zeros(7)
Sb[:3] = S_0[:3] 
Su[3:] = S_0[3:]

S1_0 = Sb * 0.6
S2_0 = Sb * 0.2
S3_0 = Sb * 0.1
S4_0 = Sb * 0.1
S5_0 = Su 



# ####### PREDICTION

df3   = file.parse(3) 
init  = df3.values[0:, 1:8] # init values

df4  = file.parse(41) 
sea  = df4.values[104:, 1:8] # seasonal infection (fitted)

df6  = file.parse(40) 
ser  = df6.values[52:, 20:27] # seasonal hospital (fitted)

ft  = np.arange(0, 52*7, 7) # time a year

init_vals = init

gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gRv, gNs, gNh = init_vals   
    
for i, j in enumerate(ft):
    
    par = np.array([0.15142, 0.06557, 0.00562, 0.0005, 0.000030, 0.00004, 0.00008]) * 1 # 0.8 #1.2  # last fitting paramter
    par = sea[i] * par
    
    rho = np.array([0.0052341, 0.0359557, 0.0052, 0.1199, 0.01704, 0.01506, 0.001303]) * 1 # 0.8 #1.2  # last fitting paramter
    rho = ser[i] * rho
    
    # params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho
    
    dummyt = np.linspace(0, 7, 8)
    nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt, rho)
    init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
        
    gS1 = np.vstack((gS1, nS1[1:,:]))
    gE1 = np.vstack((gE1, nE1[1:,:]))
    gI1 = np.vstack((gI1, nI1[1:,:]))
    gH1 = np.vstack((gH1, nH1[1:,:]))
    gR1 = np.vstack((gR1, nR1[1:,:]))
    gS2 = np.vstack((gS2, nS2[1:,:]))
    gE2 = np.vstack((gE2, nE2[1:,:]))
    gI2 = np.vstack((gI2, nI2[1:,:]))
    gH2 = np.vstack((gH2, nH2[1:,:]))
    gR2 = np.vstack((gR2, nR2[1:,:]))
    gS3 = np.vstack((gS3, nS3[1:,:]))
    gE3 = np.vstack((gE3, nE3[1:,:]))
    gI3 = np.vstack((gI3, nI3[1:,:]))
    gH3 = np.vstack((gH3, nH3[1:,:]))
    gR3 = np.vstack((gR3, nR4[1:,:]))
    gS4 = np.vstack((gS4, nS4[1:,:]))
    gE4 = np.vstack((gE4, nE4[1:,:]))
    gI4 = np.vstack((gI4, nI4[1:,:]))
    gH4 = np.vstack((gH4, nH4[1:,:]))
    gR4 = np.vstack((gR4, nR4[1:,:]))
    gS5 = np.vstack((gS5, nS5[1:,:]))
    gE5 = np.vstack((gE5, nE5[1:,:]))
    gI5 = np.vstack((gI5, nI5[1:,:]))
    gH5 = np.vstack((gH5, nH5[1:,:]))
    gR5 = np.vstack((gR5, nR5[1:,:]))
    gD  = np.vstack((gD, nD[1:,:]))
    gV  = np.vstack((gV, nV[1:,:]))
    gEv = np.vstack((gEv, nEv[1:,:]))
    gIv = np.vstack((gIv, nIv[1:,:]))
    gRv = np.vstack((gRv, nRv[1:,:]))
    gNs = np.vstack((gNs, nNs[1:,:]))
    gNh = np.vstack((gNh, nNh[1:,:]))
   
zN2  = gNs[::7]*N
zN2s = np.zeros((52,7))
for i in range(51):
    zN2s[i+1] = zN2[i+1]-zN2[i]
    
zH2  = gNh[::7]*N
zN2h = np.zeros((52,7))
for i in range(51):
    zN2h[i+1] = zH2[i+1]-zH2[i]



# CONFIDENCE INTERVALS
# num = 1000
# simulations = np.zeros((num, 52*7+1, 7))
# quantiles = [0.025, 0.975]

# for i in range(num):
#     gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gRv, gNs, gNh = init   
#     init_vals   = init # S_0, E_0, I_0, H_0, R_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
    
#     opp = np.array([0.15142, 0.06557, 0.00562, 0.0005, 0.000030, 0.00004, 0.00008])
#     opp = np.random.normal(opp, opp*0.05, 7)
#     opp[opp<0] = 0
    
#     for j in range(52):
#         dummyt = np.linspace(0, 7, 8)
        
#         opp = opp * sea[j] 
        
#         rho = np.array([0.0052341, 0.0359557, 0.0052, 0.1199, 0.01704, 0.01506, 0.001303]) 
#         rho = ser[j] * rho
            
#         nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt, rho)
#         init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
            
#         gS1 = np.vstack((gS1, nS1[1:,:]))
#         gE1 = np.vstack((gE1, nE1[1:,:]))
#         gI1 = np.vstack((gI1, nI1[1:,:]))
#         gH1 = np.vstack((gH1, nH1[1:,:]))
#         gR1 = np.vstack((gR1, nR1[1:,:]))
#         gS2 = np.vstack((gS2, nS2[1:,:]))
#         gE2 = np.vstack((gE2, nE2[1:,:]))
#         gI2 = np.vstack((gI2, nI2[1:,:]))
#         gH2 = np.vstack((gH2, nH2[1:,:]))
#         gR2 = np.vstack((gR2, nR2[1:,:]))
#         gS3 = np.vstack((gS3, nS3[1:,:]))
#         gE3 = np.vstack((gE3, nE3[1:,:]))
#         gI3 = np.vstack((gI3, nI3[1:,:]))
#         gH3 = np.vstack((gH3, nH3[1:,:]))
#         gR3 = np.vstack((gR3, nR4[1:,:]))
#         gS4 = np.vstack((gS4, nS4[1:,:]))
#         gE4 = np.vstack((gE4, nE4[1:,:]))
#         gI4 = np.vstack((gI4, nI4[1:,:]))
#         gH4 = np.vstack((gH4, nH4[1:,:]))
#         gR4 = np.vstack((gR4, nR4[1:,:]))
#         gS5 = np.vstack((gS5, nS5[1:,:]))
#         gE5 = np.vstack((gE5, nE5[1:,:]))
#         gI5 = np.vstack((gI5, nI5[1:,:]))
#         gH5 = np.vstack((gH5, nH5[1:,:]))
#         gR5 = np.vstack((gR5, nR5[1:,:]))
#         gD  = np.vstack((gD, nD[1:,:]))
#         gV  = np.vstack((gV, nV[1:,:]))
#         gEv = np.vstack((gEv, nEv[1:,:]))
#         gIv = np.vstack((gIv, nIv[1:,:]))
#         gRv = np.vstack((gRv, nRv[1:,:]))
#         gNs = np.vstack((gNs, nNs[1:,:]))
#         gNh = np.vstack((gNh, nNh[1:,:]))
    
#     simulations[i,:,:] = gNs*N  # Infected population

# qr = np.zeros((2, 52*7+1, 7))
# for i in range(2):
#     qr[i] = np.percentile(simulations, quantiles[i] * 100, axis=0)

# zQ = qr[:,::7]
# zNq = np.zeros((2,52,7))
# for i in range(51):
#     zNq[:,i+1] = zQ[:,i+1]-zQ[:,i]



color = ['purple', 'orange', 'green', 'cyan', 'blue', 'grey', 'red']
label = ['0-1 years', '2-4 years', '5-14 years', '15-34 years', '35-59 years', '60-79 years', '80+ years']


for i in range(7):
    plt.plot(ft[:-1]/7,zN2s[:-1,i], color=color[i], label=label[i]) 
    # plt.fill_between(ft[:-1]/7, zQ[1,:-1,i], zQ[0,:-1,i], color=color[i], alpha=0.2)
plt.xlabel('Week')
plt.ylabel('New cases')
plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
plt.show()

for i in range(7):
    plt.plot(ft[:-1]/7,zN2h[:-1,i], color=color[i], label=label[i]) 
plt.xlabel('Week')
plt.ylabel('New hospitalisation')
plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
plt.show()
