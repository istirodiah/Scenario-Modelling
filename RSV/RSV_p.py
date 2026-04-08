#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 12:24:13 2022

@author: istirodiah
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

from scipy.optimize import Bounds
bounds = Bounds(np.zeros((14)), 100*np.ones((14)))


def model(init_vals, params, opparams, t):
    S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0 = init_vals
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]
    # P1, P2, P3, P4 = period
    # c, sea, rho, sigma, theta, gamma, gammav, epsilon = params
    c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi = params
   
    dum   = opparams[:7]
    rho   = opparams[7:14]
    
    p        = np.zeros(7)
    beta     = np.zeros(7)
    p[3:]    = dum[3:]
    beta[:3] = dum[:3]
    
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
    
    print(opparams)
    print(p)
    
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
        # print(p)
        # print(S5[-1])
        # print(p*S5[-1])
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


def cost(opparams, modelparams, idata, hdata, i):
    init_vals, params, N, t = modelparams
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = model(init_vals, params, opparams, t)
        
    dum = 0
    
    dui  = sum(idata[:i+1])
    simi = Ns[7,:] * N
    p    = (simi - dui)**2 * 1./max(dui)
    
    duh  = sum(hdata[:i+1])
    simh = Nh[7,:] * N
    q    = (simh - duh)**2 * 1./max(duh)
    
    dum = dum + sum(p) + sum(q)
    return dum

# 0.72
# 0.88
# 1.23
# 1.71
# 1.00

file   = pd.ExcelFile('RSV.xlsx')
# df     = file.parse(3) 
# idata  = df.values[0:, 1:8]
# hdata  = df.values[0:, 20:27]

df1    = file.parse(17)#6 #12 #17 #20 #23
idata  = df1.values[0:, 1:8]
df2    = file.parse(18) 
hdata  = df2.values[0:, 1:8]

it = 156 #104
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters		
# c   = 10 * np.array([[3.60E-07, 3.00E-07, 1.03E-07, 2.32E-07, 6.40E-08, 3.23E-08, 0.00E+00],
#                      [3.00E-07, 9.19E-07, 1.09E-07, 8.59E-08, 6.49E-08, 4.01E-08, 2.25E-08],
#                      [1.03E-07, 1.09E-07, 4.22E-07, 7.46E-08, 7.68E-08, 3.21E-08, 1.37E-08],
#                      [2.32E-07, 8.59E-08, 7.46E-08, 2.60E-07, 1.03E-07, 3.77E-08, 2.61E-08],
#                      [6.40E-08, 6.49E-08, 7.68E-08, 1.03E-07, 1.31E-07, 5.91E-08, 3.37E-08],
#                      [3.23E-08, 4.01E-08, 3.21E-08, 3.77E-08, 5.91E-08, 1.19E-07, 6.09E-08],
#                      [0.00E+00, 2.25E-08, 1.37E-08, 2.61E-08, 3.37E-08, 6.09E-08, 8.79E-08]])


# c   = 10 * np.array([[2.60E-07, 2.00E-07, 9.03E-08, 1.32E-07, 5.40E-08, 2.23E-08, 0.00E+00],
#                       [2.00E-07, 9.19E-07, 1.09E-07, 8.59E-08, 6.49E-08, 4.01E-08, 2.25E-08],
#             `         [9.03E-08, 1.09E-07, 4.22E-07, 7.46E-08, 7.68E-08, 3.21E-08, 1.37E-08],
#                       [1.32E-07, 8.59E-08, 7.46E-08, 2.60E-07, 1.03E-07, 3.77E-08, 2.61E-08],
#                       [5.40E-08, 6.49E-08, 7.68E-08, 1.03E-07, 1.31E-07, 5.91E-08, 3.37E-08],
#                       [2.23E-08, 4.01E-08, 3.21E-08, 3.77E-08, 5.91E-08, 1.19E-07, 6.09E-08],
#                       [0.00E+00, 2.25E-08, 1.37E-08, 2.61E-08, 3.37E-08, 6.09E-08, 8.79E-08]])


c   = 10 * np.array([[1.046E-07, 8.024E-08, 6.986E-08, 6.114E-08, 4.288E-08, 1.957E-08, 0.000E+00],
                      [8.024E-08, 3.687E-07, 8.406E-08, 3.989E-08, 5.159E-08, 3.517E-08, 7.997E-08],
                      [6.986E-08, 8.406E-08, 5.721E-08, 2.569E-08, 2.176E-08, 1.171E-08, 2.647E-08],
                      [6.114E-08, 3.989E-08, 2.569E-08, 2.942E-08, 1.908E-08, 1.274E-08, 4.122E-08],
                      [4.288E-08, 5.159E-08, 2.176E-08, 1.908E-08, 1.418E-08, 8.350E-09, 1.447E-08],
                      [1.957E-08, 3.517E-08, 1.171E-08, 1.274E-08, 8.350E-09, 1.488E-08, 1.441E-08],
                      [0.000E+00, 7.997E-08, 2.647E-08, 4.122E-08, 1.447E-08, 1.441E-08, 2.858E-08]])


sea     = 1#np.array([0.4, 0.4, 0.8, 0.8, 0.9, 0.9])
# rho     = np.array([0.75, 0.66, 0.13, 0.33, 0.09, 0.13, 0.18])
theta   = 1./10 #np.array([0.20, 0.16, 0.06, 0.02, 0.00, 0.01, 0.04])
sigma   = theta
gamma   = 1./30
gammav  = 1./30
mu      = 1./90
alpha   = 1./7
eta     = 1./5
phi     = eta
epsilon = np.array([0.05, 0.05, 0, 0, 0, 0, 0])
# period  = P1, P2, P3, P4
params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi 

N = 83166711         # worldometer data

# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.2365, 0.3709, 0.2057, 0.0428])
N    = N * np.array([0.0176, 0.0264, 0.1001, 0.002365, 0.003709, 0.002057, 0.000428])
Ns_0 = idata[0]/N
Nh_0 = hdata[0]/N

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

########### 2021-2022 ###############
df            = file.parse(27)
init_vals     = df.values[0:32, 1:8]
init_vals[-2] = idata[156]/N
init_vals[-1] = hdata[156]/N
idata         = idata[156:]
hdata         = hdata[156:]
Ns_0 = idata[0]/N
Nh_0 = hdata[0]/N
####################################

init_vals   = S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
modelparams = init_vals, params, N, t

opparams    = np.ones(14)*0.001
opp         = np.ones((len(times), 14))*0.001

# dum = 52
# ft  = np.linspace(0, dum, int(dum/dt)+1)
# beta = np.array([0.4, 0.4, 0.2, 0.2, 0.3, 0.4])
# S, E, I, H, R, D, V, Ev, Iv, Rv, Ns, Nh = model(init_vals, period, params, beta, ft)  

# for i in range(6):
#     plt.plot(ft, Ns[:,i]*N[i], color=color[i], label=label[i])
# plt.xlabel('Week')
# plt.ylabel('New Cases')
# plt.legend(loc='best', fontsize='medium')
# plt.show()

S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]       

# for i, j in enumerate(times):
#     if i > 0:
#         dummyt = np.linspace(j, j+7, 8)
#         modelparams = init_vals, params, N, dummyt
        
#         optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, hdata, i), tol=1e-10, bounds=bounds)
#         # optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, hdata, i), tol=1e-10)
#         opp[i] = optimizer.x
    
#         # if np.any(opp[i]<0) == True:
#         #     # print("masuk")
#         #     # optimizer = opt.minimize(cost, opp[i], args=(modelparams, idata, i), tol=1e-10)
#         #     optimizer = opt.minimize(cost, opp[i], args=(modelparams, idata, hdata, i), tol=1e-10)
#         #     opp[i] = optimizer.x
            
#         nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, opp[i], dummyt)
#         init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
        
#         S1 = np.vstack((S1, nS1[1:,:]))
#         E1 = np.vstack((E1, nE1[1:,:]))
#         I1 = np.vstack((I1, nI1[1:,:]))
#         H1 = np.vstack((H1, nH1[1:,:]))
#         R1 = np.vstack((R1, nR1[1:,:]))
#         S2 = np.vstack((S2, nS2[1:,:]))
#         E2 = np.vstack((E2, nE2[1:,:]))
#         I2 = np.vstack((I2, nI2[1:,:]))
#         H2 = np.vstack((H2, nH2[1:,:]))
#         R2 = np.vstack((R2, nR2[1:,:]))
#         S3 = np.vstack((S3, nS3[1:,:]))
#         E3 = np.vstack((E3, nE3[1:,:]))
#         I3 = np.vstack((I3, nI3[1:,:]))
#         H3 = np.vstack((H3, nH3[1:,:]))
#         R3 = np.vstack((R3, nR4[1:,:]))
#         S4 = np.vstack((S4, nS4[1:,:]))
#         E4 = np.vstack((E4, nE4[1:,:]))
#         I4 = np.vstack((I4, nI4[1:,:]))
#         H4 = np.vstack((H4, nH4[1:,:]))
#         R4 = np.vstack((R4, nR4[1:,:]))
#         S5 = np.vstack((S5, nS5[1:,:]))
#         E5 = np.vstack((E5, nE5[1:,:]))
#         I5 = np.vstack((I5, nI5[1:,:]))
#         H5 = np.vstack((H5, nH5[1:,:]))
#         R5 = np.vstack((R5, nR5[1:,:]))
#         D  = np.vstack((D, nD[1:,:]))
#         V  = np.vstack((V, nV[1:,:]))
#         Ev = np.vstack((Ev, nEv[1:,:]))
#         Iv = np.vstack((Iv, nIv[1:,:]))
#         Rv = np.vstack((Rv, nRv[1:,:]))
#         Ns = np.vstack((Ns, nNs[1:,:]))
#         Nh = np.vstack((Nh, nNh[1:,:]))

# zN  = Ns[::7]*N
# zNs = np.zeros((157,7))
# for i in range(156):
#     zNs[i] = zN[i+1]-zN[i]

df3  = file.parse(29)#6 #12 #17 #20 #23
sea  = df3.values[0:, 1:8]

ft  = np.arange(0, 104*7, 7)
# sea = np.ones(7) #np.array([1.27, 1.22, 1.10, 1.09, 0.58, 0.11])
par = np.array([0.21039042, 0.25115401, 0.09320117, 0.104166667, 0.048888889, 0.084210526, 0.029411765, 0.124027, 0.0010202, 0.0610184, 0.351572, 0.00102379, 0.00100488, 0.00100488])
init_vals = S1[-1], E1[-1], I1[-1], H1[-1], R1[-1], S2[-1], E2[-1], I2[-1], H2[-1], R2[-1], S3[-1], E3[-1], I3[-1], H3[-1], R3[-1], S4[-1], E4[-1], I4[-1], H4[-1], R4[-1], S5[-1], E5[-1], I5[-1], H5[-1], R5[-1], D[-1], V[-1], Ev[-1], Iv[-1], Rv[-1], Ns[-1], Nh[-1]
fS1, fE1, fI1, fH1, fR1, fS2, fE2, fI2, fH2, fR2, fS3, fE3, fI3, fH3, fR3, fS4, fE4, fI4, fH4, fR4, fS5, fE5, fI5, fH5, fR5, fD, fV, fEv, fIv, fRv, fNs, fNh = init_vals   
    
for i, j in enumerate(ft):
    par = np.array([0.21039042, 0.25115401, 0.09320117, 0.104166667, 0.048888889, 0.084210526, 0.029411765, 0.124027, 0.0010202, 0.0610184, 0.351572, 0.00102379, 0.00100488, 0.00100488])
    if i == 357:
        par[3:7] = np.array([0.0703125, 0.073587385	, 0.051044084, 0.051724138])
    par[:7] = sea[i] * par[:7]
    # print(par)
    dummyt = np.linspace(0, 7, 8)
    nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt)
    init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
        
    fS1 = np.vstack((fS1, nS1[1:,:]))
    fE1 = np.vstack((fE1, nE1[1:,:]))
    fI1 = np.vstack((fI1, nI1[1:,:]))
    fH1 = np.vstack((fH1, nH1[1:,:]))
    fR1 = np.vstack((fR1, nR1[1:,:]))
    fS2 = np.vstack((fS2, nS2[1:,:]))
    fE2 = np.vstack((fE2, nE2[1:,:]))
    fI2 = np.vstack((fI2, nI2[1:,:]))
    fH2 = np.vstack((fH2, nH2[1:,:]))
    fR2 = np.vstack((fR2, nR2[1:,:]))
    fS3 = np.vstack((fS3, nS3[1:,:]))
    fE3 = np.vstack((fE3, nE3[1:,:]))
    fI3 = np.vstack((fI3, nI3[1:,:]))
    fH3 = np.vstack((fH3, nH3[1:,:]))
    fR3 = np.vstack((fR3, nR4[1:,:]))
    fS4 = np.vstack((fS4, nS4[1:,:]))
    fE4 = np.vstack((fE4, nE4[1:,:]))
    fI4 = np.vstack((fI4, nI4[1:,:]))
    fH4 = np.vstack((fH4, nH4[1:,:]))
    fR4 = np.vstack((fR4, nR4[1:,:]))
    fS5 = np.vstack((fS5, nS5[1:,:]))
    fE5 = np.vstack((fE5, nE5[1:,:]))
    fI5 = np.vstack((fI5, nI5[1:,:]))
    fH5 = np.vstack((fH5, nH5[1:,:]))
    fR5 = np.vstack((fR5, nR5[1:,:]))
    fD  = np.vstack((fD, nD[1:,:]))
    fV  = np.vstack((fV, nV[1:,:]))
    fEv = np.vstack((fEv, nEv[1:,:]))
    fIv = np.vstack((fIv, nIv[1:,:]))
    fRv = np.vstack((fRv, nRv[1:,:]))
    fNs = np.vstack((fNs, nNs[1:,:]))
    fNh = np.vstack((fNh, nNh[1:,:]))
   
zN  = fNs[::7]*N
zNs = np.zeros((104,7))
for i in range(103):
    zNs[i] = zN[i+1]-zN[i]

# zi = np.zeros((105,7))
# for i in range(105):
#     zi[i] = sum(idata[:i+1])

# color = ['purple', 'orange', 'green', 'cyan', 'blue', 'grey', 'red']
# label = ['0-1 years', '2-4 years', '5-14 years', '15-34 years', '35-59 years', '60-79 years', '80+ years']

# for i in range(7):
#     plt.plot(np.linspace(1, 105, 105),Ns[::7,i]*N[i], color=color[i], label=label[i]) 
#     plt.plot(np.linspace(1, 105, 105), zi[:,i],'o', color=color[i])
    
    
# zN  = fNs[::7]*N
# zNs = np.zeros((5,6))
# for i in range(5):
#     zNs[i] = zN[i+1]-zN[i]
    
# for i in range(6):
#     plt.plot(np.linspace(1, 47, 47), idata[:,i], color=color[i])
#     # plt.plot(np.linspace(47, 48), [idata[-1,i], zNs[0,i]], color=color[i])
#     plt.plot(np.linspace(48, 53, 5),zNs[:,i], color=color[i])

# import csv
# with open('file.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerows(init_vals)