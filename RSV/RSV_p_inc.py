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
bounds = Bounds(np.zeros((7)), 100*np.ones((7)))


def model(init_vals, params, opparams, t):
    S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0 = init_vals
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]
    # P1, P2, P3, P4 = period
    # c, sea, rho, sigma, theta, gamma, gammav, epsilon = params
    c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho = params
    # print(rho)
    
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
        # next_Nh  = (rho1*I1[-1] + rho2*I2[-1] + rho3*I3[-1] + rho4*I4[-1] + rho5*I5[-1])*dt
        
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
    
    # duh  = sum(hdata[:i+1])
    # simh = Nh[7,:] * N
    # q    = (simh - duh)**2 * 1./max(duh)
    
    dum = dum + sum(p) #+ sum(q)
    return dum


file   = pd.ExcelFile('RSV_24.xlsx')
file1  = pd.ExcelFile('RKI_ARE.xlsx')
# df     = file.parse(3) 
# idata  = df.values[0:, 1:8]
# hdata  = df.values[0:, 20:27]

df1    = file.parse(55)#30 #12 #17 #20 #23
idata  = df1.values[209:, 2:9] #209
df2    = file.parse(34) 
hdata  = df2.values[0:, 2:9]
df1    = file1.parse(2)#30 #12 #17 #20 #23
idata  = df1.values[1:, 17:24] #209

it = 156 #365 #
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters		
# c   = 10 * np.array([[3.60E-07, 3.00E-07, 1.03E-07, 2.32E-07, 6.40E-08, 3.23E-08, 0.00E+00],
#                       [3.00E-07, 9.19E-07, 1.09E-07, 8.59E-08, 6.49E-08, 4.01E-08, 2.25E-08],
#                       [1.03E-07, 1.09E-07, 4.22E-07, 7.46E-08, 7.68E-08, 3.21E-08, 1.37E-08],
#                       [2.32E-07, 8.59E-08, 7.46E-08, 2.60E-07, 1.03E-07, 3.77E-08, 2.61E-08],
#                       [6.40E-08, 6.49E-08, 7.68E-08, 1.03E-07, 1.31E-07, 5.91E-08, 3.37E-08],
#                       [3.23E-08, 4.01E-08, 3.21E-08, 3.77E-08, 5.91E-08, 1.19E-07, 6.09E-08],
#                       [0.00E+00, 2.25E-08, 1.37E-08, 2.61E-08, 3.37E-08, 6.09E-08, 8.79E-08]])


# c   = 10 * np.array([[2.60E-07, 2.00E-07, 9.03E-08, 1.32E-07, 5.40E-08, 2.23E-08, 0.00E+00],
#                       [2.00E-07, 9.19E-07, 1.09E-07, 8.59E-08, 6.49E-08, 4.01E-08, 2.25E-08],
#                       [9.03E-08, 1.09E-07, 4.22E-07, 7.46E-08, 7.68E-08, 3.21E-08, 1.37E-08],
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
rho     = np.array([0.25, 0.15, 0.08, 0.06, 0.05, 0.05, 0.05])
theta   = 1./10 #np.array([0.20, 0.16, 0.06, 0.02, 0.00, 0.01, 0.04])
sigma   = 0#theta
gamma   = 1./30
gammav  = 1./30
mu      = 1./90
alpha   = 1./7
eta     = 1./5
phi     = 0#eta
epsilon = np.array([0.05, 0.05, 0, 0, 0, 0, 0])
# period  = P1, P2, P3, P4
params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho

N = 83166711         # worldometer data

# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.2365, 0.3709, 0.2057, 0.0428])
N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.3709, 0.2057, 0.0428])

# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.03709, 0.02057, 0.00428]) #ARE, RKI & Muspad

Ns_0 = idata[0]/N #np.array([0.08184637874068951, 0.026686680590009976, 0.001963659362269918, 0.006165203598712996, 0.006500782430498748, 0.009699830065996609, 0.025796133757925076])
Nh_0 = np.zeros(7)#hdata[0]/N

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

# week 2020 ARE
# init_vals = (np.array([1.01656724e-04, 1.48363100e-04, 5.79339474e-01, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([5.36484470e-06, 5.22211914e-06, 3.00627645e-04, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([2.24181885e-06, 3.34206833e-06, 2.56370842e-04, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.79051575e-06, 1.67058151e-06, 6.44779134e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.00043569, 0.00020905, 0.0018915 , 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.39109803, 0.497735  , 0.20290176, 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([8.21429648e-03, 6.44365715e-03, 7.86001079e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([4.14468380e-03, 5.30149245e-03, 8.36374134e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([2.58453941e-03, 2.18629778e-03, 1.75869601e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.02813156, 0.02092523, 0.0005547 , 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.20822023, 0.16348842, 0.10058498, 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([2.76754108e-03, 1.34520829e-03, 2.60131198e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.89903445e-03, 1.50460400e-03, 3.56111129e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([8.96507982e-04, 4.58833516e-04, 5.57710287e-06, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.00945752, 0.00452892, 0.00021801, 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.10377627, 0.09971641, 0.10002636, 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([7.10526706e-04, 4.28669781e-04, 1.29496211e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([7.54600640e-04, 6.92689686e-04, 2.37336566e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([2.07526145e-04, 1.16280469e-04, 2.09028721e-06, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.01426362, 0.00732267, 0.00066778, 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.030897  , 0.01375706, 0.00126259, 0.97327893, 0.95463484,
#         0.98194327, 0.97614372]),
#  np.array([0.        , 0.        , 0.        , 0.00104163, 0.00131796,
#         0.00194411, 0.00213153]),
#  np.array([0.        , 0.        , 0.        , 0.00233098, 0.00287869,
#         0.00379722, 0.00446324]),
#  np.array([0.        , 0.        , 0.        , 0.00016703, 0.00017567,
#         0.00020899, 0.00025156]),
#  np.array([0.        , 0.        , 0.        , 0.02085097, 0.03500838,
#         0.01049487, 0.01436913]),
#  np.array([0.1170148 , 0.06700359, 0.01290816, 0.01126199, 0.01836037,
#         0.00497454, 0.00813472]),
#  np.array([0.08691385, 0.10733486, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.00138782, 0.00109183, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.00318595, 0.00284516, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.01382436, 0.00916252, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  np.array([0.40797235282291755, 0.2344360446577506, 0.028504269082026645,
#         0.11839165265261818, 0.1939604998492128, 0.05730723199909318,
#         0.09083431820826231], dtype=object),
#  np.array([0.14606109, 0.06305509, 0.00750641, 0.00595435, 0.0082506 ,
#         0.002242  , 0.00366259]))
# init_vals   = S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
# modelparams = init_vals, params, N, t

########### 2020-2021 ###############
# df            = file.parse(27)
# init_vals     = df.values[0:32, 1:8]
# init_vals[-1] = idata[156]/N
# idata         = idata[156:]
# hdata         = hdata[156:]
# Ns_0 = idata[0]/N
# Nh_0 = hdata[0]/N
# S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0 = init_vals    
####################################

########### 2021-2022 ###############
# df            = file.parse(27)
# init_vals     = df.values[0:32, 10:17]
# init_vals[-1] = idata[209]/N
# idata         = idata[209:]
# hdata         = hdata[209:]
# Ns_0 = idata[0]/N
# Nh_0 = hdata[0]/N#np.zeros(7)#hdata[0]/N
# S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0 = init_vals    
####################################


#### init run 2021 RKI & MuSPAD
init_vals = (np.array([1.45534666e-05, 1.60037366e-05, 5.98572249e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.61310233e-09, 1.01721254e-09, 8.91701798e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([7.10418407e-10, 4.82818328e-10, 7.20417522e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([6.34853680e-10, 2.16455136e-10, 1.97955575e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.52919015e-05, 4.99775715e-06, 3.90495681e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.66090538, 0.70355637, 0.20031736, 0.        , 0.        , 0.        , 0.        ]),
  np.array([1.28755885e-05, 1.71104120e-05, 2.23696467e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([6.15600106e-06, 6.26977537e-06, 2.52898989e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([3.63352910e-06, 1.38132096e-06, 6.54526176e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.83657263e-03, 6.50670912e-04, 1.14848905e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.13621771, 0.11200039, 0.10007359, 0.        , 0.        , 0.        , 0.        ]),
  np.array([1.77517317e-06, 1.82517662e-06, 7.45164513e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.16280396e-06, 8.02886850e-07, 1.41908642e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([5.63759728e-07, 1.50188326e-07, 3.21024719e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([2.74936161e-03, 8.78054896e-04, 4.76215056e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.10120947, 0.10031037, 0.10001519, 0.        , 0.        , 0.        , 0.        ]),
  np.array([6.69239799e-07, 8.19107402e-07, 3.72423518e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([7.54640126e-07, 5.10272441e-07, 1.73284632e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([2.57653780e-07, 7.56867469e-08, 2.59405488e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([2.74936161e-03, 8.78054896e-04, 4.76215056e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([9.88325109e-03, 3.10963628e-03, 1.72683818e-04, 9.99451957e-01, 9.99318771e-01, 9.99046136e-01, 9.97383339e-01]),
  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.24074805e-07, 3.07023398e-07, 1.02511415e-05, 3.47727485e-06]),
  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.22219311e-06, 5.63944697e-06, 9.28331238e-06, 3.14346923e-05]),
  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.69430692e-07, 6.05628114e-07, 7.81097077e-07, 3.06259343e-06]),
  np.array([0.        , 0.        , 0.        , 0.00041594, 0.00046934, 0.0006274 , 0.00176382]),
  np.array([0.02578501, 0.00844977, 0.00091853, 0.00060541, 0.00062892, 0.00093747, 0.00249495]),
  np.array([0.07004251, 0.07370819, 0.        , 0.        , 0.        , 0.        , 0.        ]),
  np.array([1.11385836e-06, 1.29094602e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.21832352e-05, 4.57290487e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.00087313, 0.00029674, 0.        , 0.        , 0.        , 0.        , 0.        ]),
  np.array([0., 0., 0., 0., 0., 0., 0.]),
  np.array([0., 0., 0., 0., 0., 0., 0.]))
  # np.array([0.08184637874068951, 0.026686680590009976, 0.001963659362269918, 0.006165203598712996, 0.006500782430498748, 0.009699830065996609, 0.025796133757925076], dtype=object),
  # np.array([0.03164966899408866, 0.0078009251883809725, 0.000532516785945772, 0.0003197666744036966, 0.00028266528073324506, 0.00042219721920451193, 0.0011262251603684442], dtype=object))
  # np.array([1.279598314839203e-05, 1.5185181583701212e-06, 8.425222039925888e-08, 4.41741461080723e-08, 1.5351293804979955e-08, 7.180109105042209e-08, 3.3705407841185605e-07], dtype=object))
# init_vals[30] = np.zeros(7)
# init_vals[31] = np.zeros(7)

opparams    = np.ones(7)*0.001
opp         = np.ones((len(times), 7))*0.001

# init_vals   = S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
# modelparams = init_vals, params, N, t

S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0 = init_vals    
S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]       

for i, j in enumerate(times):
    if i > 0:
        dummyt = np.linspace(j, j+7, 8)
        modelparams = init_vals, params, N, dummyt
        
        optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, i), tol=1e-10, bounds=bounds)
        # optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, hdata, i), tol=1e-10)
        opp[i] = optimizer.x
    
        # if np.any(opp[i]<0) == True:
        #     # print("masuk")
        #     # optimizer = opt.minimize(cost, opp[i], args=(modelparams, idata, i), tol=1e-10)
        #     optimizer = opt.minimize(cost, opp[i], args=(modelparams, idata, hdata, i), tol=1e-10)
        #     opp[i] = optimizer.x
            
        nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, opp[i], dummyt)
        init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
        
        S1 = np.vstack((S1, nS1[1:,:]))
        E1 = np.vstack((E1, nE1[1:,:]))
        I1 = np.vstack((I1, nI1[1:,:]))
        H1 = np.vstack((H1, nH1[1:,:]))
        R1 = np.vstack((R1, nR1[1:,:]))
        S2 = np.vstack((S2, nS2[1:,:]))
        E2 = np.vstack((E2, nE2[1:,:]))
        I2 = np.vstack((I2, nI2[1:,:]))
        H2 = np.vstack((H2, nH2[1:,:]))
        R2 = np.vstack((R2, nR2[1:,:]))
        S3 = np.vstack((S3, nS3[1:,:]))
        E3 = np.vstack((E3, nE3[1:,:]))
        I3 = np.vstack((I3, nI3[1:,:]))
        H3 = np.vstack((H3, nH3[1:,:]))
        R3 = np.vstack((R3, nR4[1:,:]))
        S4 = np.vstack((S4, nS4[1:,:]))
        E4 = np.vstack((E4, nE4[1:,:]))
        I4 = np.vstack((I4, nI4[1:,:]))
        H4 = np.vstack((H4, nH4[1:,:]))
        R4 = np.vstack((R4, nR4[1:,:]))
        S5 = np.vstack((S5, nS5[1:,:]))
        E5 = np.vstack((E5, nE5[1:,:]))
        I5 = np.vstack((I5, nI5[1:,:]))
        H5 = np.vstack((H5, nH5[1:,:]))
        R5 = np.vstack((R5, nR5[1:,:]))
        D  = np.vstack((D, nD[1:,:]))
        V  = np.vstack((V, nV[1:,:]))
        Ev = np.vstack((Ev, nEv[1:,:]))
        Iv = np.vstack((Iv, nIv[1:,:]))
        Rv = np.vstack((Rv, nRv[1:,:]))
        Ns = np.vstack((Ns, nNs[1:,:]))
        Nh = np.vstack((Nh, nNh[1:,:]))

zN  = Ns[::7]*N
zNs = np.zeros((it,7))
for i in range(it-1):
    zNs[i] = zN[i+1]-zN[i]
    
zR  = R5[::7]*N
zS  = S5[::7]*N

# df3  = file.parse(29)#6 #12 #17 #20 #23
# sea  = df3.values[0:, 1:8]

# ft  = np.arange(0, 104*7, 7)

# df4  = file.parse(26)#6 #12 #17 #20 #23
# pr   = df4.values[209:, 34:41]#1:8]

# df5 = file.parse(39)#6 #12 #17 #20 #23
# r   = df5.values[209:313, 28:35]#1:8]
# # pr  = df5.values[209:313, 21:28]#1:8]

# # init_vals = S1[-1], E1[-1], I1[-1], H1[-1], R1[-1], S2[-1], E2[-1], I2[-1], H2[-1], R2[-1], S3[-1], E3[-1], I3[-1], H3[-1], R3[-1], S4[-1], E4[-1], I4[-1], H4[-1], R4[-1], S5[-1], E5[-1], I5[-1], H5[-1], R5[-1], D[-1], V[-1], Ev[-1], Iv[-1], Rv[-1], Ns[-1], Nh[-1]
# fS1, fE1, fI1, fH1, fR1, fS2, fE2, fI2, fH2, fR2, fS3, fE3, fI3, fH3, fR3, fS4, fE4, fI4, fH4, fR4, fS5, fE5, fI5, fH5, fR5, fD, fV, fEv, fIv, fRv, fNs, fNh = init_vals   

# # for i, j in enumerate(ft):
# #     params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, r[i]
    
# for i, j in enumerate(ft):
#     # par = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058*1]) #RKI
    
#     par = np.array([0.70947/1, 0.33580/1, 0.01652/1, 0.0941/30, 0.0576/30, 0.0537/30, 0.0326/45]) #Muspad
#     if i >= 52:
#         par = np.array([0.70947/1, 0.33580/1, 0.01652/1, 0.2075/60, 0.0976/60, 0.1360/60, 0.1667/90])
#     par = sea[i] * par
    
#     # params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, r[i]
    
#     # print(i)
#     # print(rho)
#     # print(r[i])
#     # print(" ")
    
#     # par = pr[i]
    
#     dummyt = np.linspace(0, 7, 8)
#     nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt, par)
#     # nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt, r[i])
#     init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
        
#     fS1 = np.vstack((fS1, nS1[1:,:]))
#     fE1 = np.vstack((fE1, nE1[1:,:]))
#     fI1 = np.vstack((fI1, nI1[1:,:]))
#     fH1 = np.vstack((fH1, nH1[1:,:]))
#     fR1 = np.vstack((fR1, nR1[1:,:]))
#     fS2 = np.vstack((fS2, nS2[1:,:]))
#     fE2 = np.vstack((fE2, nE2[1:,:]))
#     fI2 = np.vstack((fI2, nI2[1:,:]))
#     fH2 = np.vstack((fH2, nH2[1:,:]))
#     fR2 = np.vstack((fR2, nR2[1:,:]))
#     fS3 = np.vstack((fS3, nS3[1:,:]))
#     fE3 = np.vstack((fE3, nE3[1:,:]))
#     fI3 = np.vstack((fI3, nI3[1:,:]))
#     fH3 = np.vstack((fH3, nH3[1:,:]))
#     fR3 = np.vstack((fR3, nR4[1:,:]))
#     fS4 = np.vstack((fS4, nS4[1:,:]))
#     fE4 = np.vstack((fE4, nE4[1:,:]))
#     fI4 = np.vstack((fI4, nI4[1:,:]))
#     fH4 = np.vstack((fH4, nH4[1:,:]))
#     fR4 = np.vstack((fR4, nR4[1:,:]))
#     fS5 = np.vstack((fS5, nS5[1:,:]))
#     fE5 = np.vstack((fE5, nE5[1:,:]))
#     fI5 = np.vstack((fI5, nI5[1:,:]))
#     fH5 = np.vstack((fH5, nH5[1:,:]))
#     fR5 = np.vstack((fR5, nR5[1:,:]))
#     fD  = np.vstack((fD, nD[1:,:]))
#     fV  = np.vstack((fV, nV[1:,:]))
#     fEv = np.vstack((fEv, nEv[1:,:]))
#     fIv = np.vstack((fIv, nIv[1:,:]))
#     fRv = np.vstack((fRv, nRv[1:,:]))
#     fNs = np.vstack((fNs, nNs[1:,:]))
#     fNh = np.vstack((fNh, nNh[1:,:]))
   
# zN  = fNs[::7]*N
# zNs = np.zeros((104,7))
# for i in range(103):
#     zNs[i+1] = zN[i+1]-zN[i]
    
# zH  = fNh[::7]*N
# zNh = np.zeros((104,7))
# for i in range(103):
#     zNh[i+1] = zH[i+1]-zH[i]

# ####### PREDICTION

# # ##### init ARE
# init = (np.array([5.25954214e-12, 2.93287674e-08, 5.20702398e-01, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([2.42908027e-10, 1.51207216e-09, 9.02770230e-04, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([1.12923148e-10, 1.02521309e-09, 8.79545294e-04, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([1.09172984e-10, 5.78976088e-10, 2.55449709e-04, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([2.68229358e-06, 8.43968293e-07, 8.53993685e-03, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([0.00068829, 0.34508608, 0.20865999, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.00167471, 0.00402753, 0.00026312, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.00108037, 0.00371165, 0.00034656, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([9.42797973e-04, 1.83668412e-03, 8.80943062e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([0.04125139, 0.05069491, 0.00279242, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.04467454, 0.22129752, 0.10196436, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([8.83563397e-03, 1.40000721e-03, 8.62108464e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([0.00877693, 0.00195906, 0.0001646 , 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([5.93046948e-03, 7.64583635e-04, 3.26628168e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([0.10866277, 0.01777831, 0.00106777, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.10153998, 0.11038557, 0.09994059, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([6.52666427e-03, 3.65626857e-04, 4.25452798e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([0.01176876, 0.00090857, 0.00012806, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([4.40556366e-03, 2.07012601e-04, 1.48084281e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#   np.array([0.15494938, 0.02112592, 0.00267511, 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.26587525, 0.05615145, 0.00795148, 0.80557139, 0.78681198,
#         0.83407405, 0.34910309]),
#   np.array([0.        , 0.        , 0.        , 0.0089147 , 0.01176344,
#         0.01172325, 0.05246109]),
#   np.array([0.        , 0.        , 0.        , 0.03702465, 0.04380773,
#         0.0393671 , 0.16230774]),
#   np.array([0.        , 0.        , 0.        , 0.00346329, 0.003259  ,
#         0.00283814, 0.01142143]),
#   np.array([0.        , 0.        , 0.        , 0.13639061, 0.13727989,
#         0.10368453, 0.39606819]),
#   np.array([0.49083074, 0.20058827, 0.050453  , 0.04271136, 0.05291232,
#         0.0259905 , 0.0895974 ]),
#   np.array([0.00018893, 0.01345326, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.00016039, 0.00013037, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.00096079, 0.00073802, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([0.00614821, 0.00354015, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#   np.array([1.5414720441580134, 0.5604895565514856, 0.111345017197848,
#         0.48504276543580976, 0.6050313545294502, 0.3221464289504054,
#         1.149472386684592], dtype=object),
#   np.array([0.6392551563349882, 0.1877117309043025, 0.029368090828649655,
#         0.022727802343400695, 0.02388957331670383, 0.011785046980090993,
#         0.041969579174375236], dtype=object))

# df3  = file.parse(29) # 29 MUSPAD & RKI # 38 ARE
# sea  = df3.values[52:, 1:8]

# df6  = file.parse(40) 
# # ser  = df6.values[52:, 20:27] #RKI
# # ser  = df6.values[52:, 1:8] #ARE
# ser  = df6.values[52:, 46:53] #Muspad

# ft  = np.arange(0, 52*7, 7)

# init_vals = fS1[-1], fE1[-1], fI1[-1], fH1[-1], fR1[-1], fS2[-1], fE2[-1], fI2[-1], fH2[-1], fR2[-1], fS3[-1], fE3[-1], fI3[-1], fH3[-1], fR3[-1], fS4[-1], fE4[-1], fI4[-1], fH4[-1], fR4[-1], fS5[-1], fE5[-1], fI5[-1], fH5[-1], fR5[-1], fD[-1], fV[-1], fEv[-1], fIv[-1], fRv[-1], fNs[-1], fNh[-1]
# # init_vals = init
# gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gRv, gNs, gNh = init_vals   
    
# for i, j in enumerate(ft):
    
#     # par = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058]) #RKI
#     # par = np.array([0.50947, 0.13580, 0.00652, 0.00012, 0.00004, 0.00016, 0.00013]) #RKI Low
#     # par = np.array([0.80947, 0.43580, 0.02652, 0.00052, 0.00030, 0.00056, 0.00078]) #RKI Up
#     par = np.array([0.70947/2, 0.33580/2, 0.01652/2, 0.0941/20, 0.0576/20, 0.0537/20, 0.0326/20]) #Muspad
#     # par = np.array([0.50947/2, 0.23580/2, 0.01152/2, 0.0511/20, 0.0466/20, 0.0280/20, 0.0135/20]) #Muspad Low
#     # par = np.array([0.97947/2, 0.43580/2, 0.02652/2, 0.1118/20, 0.1003/20, 0.1064/20, 0.1030/20]) #Muspad Up
#     # par = np.array([0.50947, 0.133580, 0.11652, 0.032, 0.0020, 0.00036, 0.0058]) #ARE
#     # par = np.array([0.30947, 0.077160, 0.05652, 0.017, 0.001, 0.00018, 0.0013]) #ARE Low
#     # par = np.array([0.99947, 0.337160, 0.31652, 0.097, 0.004, 0.00062, 0.0153]) #ARE Up
#     par = sea[i] * par
    
#     # r = np.array([0.012341, 0.0099557, 0.0001172, 0.199, 0.0704, 0.0506, 0.0303]) #RKI
#     r = np.array([0.43341, 0.69557, 0.029917, 0.031869, 0.001143, 0.011624, 0.010319]) #Muspad
#     # r = np.array([0.43341, 0.159557, 0.0005192, 0.00099, 0.000104, 0.000106, 0.000103]) #ARE
    
#     # r = r*0.7  #low
#     r = r*1.7  #up
    
#     rho = ser[i] * r
    
#     # params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho
    
#     dummyt = np.linspace(0, 7, 8)
#     nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt, rho)
#     init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nRv[-1], nNs[-1], nNh[-1]
        
#     gS1 = np.vstack((gS1, nS1[1:,:]))
#     gE1 = np.vstack((gE1, nE1[1:,:]))
#     gI1 = np.vstack((gI1, nI1[1:,:]))
#     gH1 = np.vstack((gH1, nH1[1:,:]))
#     gR1 = np.vstack((gR1, nR1[1:,:]))
#     gS2 = np.vstack((gS2, nS2[1:,:]))
#     gE2 = np.vstack((gE2, nE2[1:,:]))
#     gI2 = np.vstack((gI2, nI2[1:,:]))
#     gH2 = np.vstack((gH2, nH2[1:,:]))
#     gR2 = np.vstack((gR2, nR2[1:,:]))
#     gS3 = np.vstack((gS3, nS3[1:,:]))
#     gE3 = np.vstack((gE3, nE3[1:,:]))
#     gI3 = np.vstack((gI3, nI3[1:,:]))
#     gH3 = np.vstack((gH3, nH3[1:,:]))
#     gR3 = np.vstack((gR3, nR4[1:,:]))
#     gS4 = np.vstack((gS4, nS4[1:,:]))
#     gE4 = np.vstack((gE4, nE4[1:,:]))
#     gI4 = np.vstack((gI4, nI4[1:,:]))
#     gH4 = np.vstack((gH4, nH4[1:,:]))
#     gR4 = np.vstack((gR4, nR4[1:,:]))
#     gS5 = np.vstack((gS5, nS5[1:,:]))
#     gE5 = np.vstack((gE5, nE5[1:,:]))
#     gI5 = np.vstack((gI5, nI5[1:,:]))
#     gH5 = np.vstack((gH5, nH5[1:,:]))
#     gR5 = np.vstack((gR5, nR5[1:,:]))
#     gD  = np.vstack((gD, nD[1:,:]))
#     gV  = np.vstack((gV, nV[1:,:]))
#     gEv = np.vstack((gEv, nEv[1:,:]))
#     gIv = np.vstack((gIv, nIv[1:,:]))
#     gRv = np.vstack((gRv, nRv[1:,:]))
#     gNs = np.vstack((gNs, nNs[1:,:]))
#     gNh = np.vstack((gNh, nNh[1:,:]))
   
# zN2  = gNs[::7]*N
# zN2s = np.zeros((52,7))
# for i in range(51):
#     zN2s[i+1] = zN2[i+1]-zN2[i]
    
# zH2  = gNh[::7]*N
# zN2h = np.zeros((52,7))
# for i in range(51):
#     zN2h[i+1] = zH2[i+1]-zH2[i]



# num = 1000
# simulations = np.zeros((num, 52*7+1, 7))
# quantiles = [0.025, 0.975]

# for i in range(num):
#     gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gRv, gNs, gNh = init   
#     init_vals   = init # S_0, E_0, I_0, H_0, R_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
    
#     # opp = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058]) #RKI
#     # opp = np.array([0.70947, 0.33580, 0.01652, 0.0703/15, 0.0836/15, 0.0466/15, 0.0517/15]) #Muspad
#     opp = np.array([7.0947, 0.67160, 0.01652, 0.017, 0.010, 0.0036, 0.0058]) #ARE
#     opp = np.random.normal(opp, 0.05, 7)
#     opp[opp<0] = 0
    
#     for j in range(52):
#         dummyt = np.linspace(0, 7, 8)
        
#         opp = opp * sea[j] 
            
#         nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt)
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

# # m_result = np.mean(simulations, axis=0)
# qr = np.zeros((2, 52*7+1, 7))
# for i in range(2):
#     qr[i] = np.percentile(simulations, quantiles[i] * 100, axis=0)

# zQ = qr[:,::7]*N
# zNq = np.zeros((2,52,7))
# for i in range(51):
#     zNq[:,i+1] = zQ[:,i+1]-zQ[:,i]


# # # zi = np.zeros((105,7))
# # # for i in range(105):
# # #     zi[i] = sum(idata[:i+1])

# # color = ['purple', 'orange', 'green', 'cyan', 'blue', 'grey', 'red']
# # label = ['0-1 years', '2-4 years', '5-14 years', '15-34 years', '35-59 years', '60-79 years', '80+ years']

# # for i in range(7):
#     plt.plot(ft[:-1]/7,zN2h[:-1,i], color=color[i], label=label[i]) 
# plt.xlabel('Week')
# plt.ylabel('Hospitalisation')
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
    
# # zN  = fNs[::7]*N
# # zNs = np.zeros((5,6))
# # for i in range(5):
# #     zNs[i] = zN[i+1]-zN[i]
    
# # for i in range(6):
# #     plt.plot(np.linspace(1, 47, 47), idata[:,i], color=color[i])
# #     # plt.plot(np.linspace(47, 48), [idata[-1,i], zNs[0,i]], color=color[i])
# #     plt.plot(np.linspace(48, 53, 5),zNs[:,i], color=color[i])

# # import csv
# # with open('file.csv', 'w', newline='') as f:
# #     writer = csv.writer(f)
# #     writer.writerows(init_vals)

###ARE
# (array([5.36632744e-10, 3.10149679e-08, 5.21328641e-01, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([4.56695504e-10, 1.45071547e-09, 1.04926203e-03, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([7.75390099e-09, 2.02470877e-08, 6.78660603e-03, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([2.33997719e-10, 1.11865822e-10, 1.11880830e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([7.07375334e-06, 2.46626253e-06, 1.54623824e-02, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([0.02008788, 0.36102767, 0.23303015, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.00277173, 0.003998  , 0.00033859, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.02334386, 0.03212248, 0.00215975, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([1.42910902e-04, 6.59735220e-05, 2.65371717e-06, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([0.08918607, 0.08125355, 0.00470152, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.20004435, 0.30141005, 0.10793365, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.00986927, 0.00180273, 0.00010558, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.073182  , 0.01369013, 0.00067701, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([2.27232910e-04, 1.76260624e-05, 5.55495756e-07, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([0.15417834, 0.02841265, 0.0014869 , 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.23417597, 0.12950039, 0.10140532, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([4.53498410e-03, 4.01822552e-04, 5.01857739e-05, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([0.03158899, 0.00309472, 0.00032361, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([4.41801201e-05, 2.02592618e-06, 1.32977016e-07, 0.00000000e+00,
#         0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  array([0.14723316, 0.02474848, 0.00314631, 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.29254266, 0.06543345, 0.00935686, 0.79584193, 0.78567801,
#         0.84220914, 0.3903893 ]),
#  array([0.        , 0.        , 0.        , 0.00965147, 0.01242071,
#         0.01250992, 0.05301831]),
#  array([0.        , 0.        , 0.        , 0.06834749, 0.07560402,
#         0.06511645, 0.26389899]),
#  array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 8.33699845e-05,
#         4.53762574e-05, 8.78007321e-05, 4.59339246e-04]),
#  array([0.        , 0.        , 0.        , 0.12800964, 0.12971086,
#         0.09570435, 0.36836746]),
#  array([0., 0., 0., 0., 0., 0., 0.]),
#  array([0.00180774, 0.01383065, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.00017082, 0.00012611, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.00155116, 0.00109606, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([0.00585227, 0.00339636, 0.        , 0.        , 0.        ,
#         0.        , 0.        ]),
#  array([1.5415696345372074, 0.5609228985426079, 0.11171886373356042,
#         0.48583662974449426, 0.6057596601218064, 0.3229873095417688,
#         1.1498560577047636], dtype=object),
#  array([0.1738208011117825, 0.021283711249924633, 0.0012503375924348666,
#         0.0012893501617204386, 0.002306261397900231, 0.010419753047035773,
#         0.05076176779755201], dtype=object))