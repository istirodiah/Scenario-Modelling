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


def model(init_vals, params, opparams, t, rho):
    S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Hv_0, Rv_0, Ns_0, Nh_0 = init_vals
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Hv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Hv_0], [Rv_0], [Ns_0], [Nh_0]
    # P1, P2, P3, P4 = period
    # c, sea, rho, sigma, theta, gamma, gammav, epsilon = params
    c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, r = params
    # print(rho)
    
    p        = np.zeros(7)
    beta     = np.zeros(7)
    p[3:]    = opparams[3:]
    beta[:3] = opparams[:3]
    pv       = p #* 0.3 # efficacy 80
    
    beta1  = beta
    beta2  = beta * 0.75
    beta3  = beta * 0.5
    beta4  = beta * 0.25
    betav  = beta 
    rho1   = rho
    rho2   = rho * 0.75
    rho3   = rho * 0.5
    rho4   = rho * 0.25
    rho5   = rho * 0.1
    rhov   = rho5 * 0.2
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
    # print(p)
    # print(betav)
    dt = (t[1] - t[0])*1./7
    
    for i in t[1:]:
        
        next_Ns  = Ns[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta1*S1[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta2*S2[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta3*S3[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta4*S4[-1] + c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*betav*V[-1] + p*S5[-1] + pv*V[-1])*dt
        next_Nh  = Nh[-1] + (rho1*I1[-1] + rho2*I2[-1] + rho3*I3[-1] + rho4*I4[-1] + rho5*I5[-1] + rhov*Iv[-1])*dt
        # next_Nh  = (rho1*I1[-1] + rho2*I2[-1] + rho3*I3[-1] + rho4*I4[-1] + rho5*I5[-1])*dt
        
        next_S1 = S1[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta1*S1[-1] + epsilon*S1[-1])*dt
        next_E1 = E1[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta1*S1[-1] - alpha*E1[-1])*dt
        next_I1 = I1[-1] + (alpha*E1[-1] - (theta + rho1 + sigma1)*I1[-1])*dt
        next_H1 = H1[-1] + (rho1*I1[-1] - (eta + phi1)*H1[-1])*dt
        next_R1 = R1[-1] + (eta*H1[-1] + theta*I1[-1] - gamma*R1[-1])*dt
        
        next_S2 = S2[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta2*S2[-1] - mu*V[-1] - gamma*R1[-1])*dt
        next_E2 = E2[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta2*S2[-1] - alpha*E2[-1])*dt
        next_I2 = I2[-1] + (alpha*E2[-1] - (theta + rho2 + sigma2)*I2[-1])*dt
        next_H2 = H2[-1] + (rho2*I2[-1] - (eta + phi2)*H2[-1])*dt
        next_R2 = R2[-1] + (eta*H2[-1] + theta*I2[-1] - gamma*R2[-1])*dt
        
        next_S3 = S3[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta3*S3[-1] - gammav*Rv[-1] - gamma*R2[-1])*dt
        next_E3 = E3[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta3*S3[-1] - alpha*E3[-1])*dt
        next_I3 = I3[-1] + (alpha*E3[-1] - (theta + rho3 + sigma3)*I3[-1])*dt
        next_H3 = H3[-1] + (rho3*I3[-1] - (eta + phi3)*H3[-1])*dt
        next_R3 = R3[-1] + (eta*H3[-1] + theta*I3[-1] - gamma*R3[-1])*dt
        
        next_S4 = S4[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta4*S4[-1] - gamma*R3[-1])*dt
        next_E4 = E4[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*beta4*S4[-1] - alpha*E4[-1])*dt
        next_I4 = I4[-1] + (alpha*E4[-1] - (theta + rho4 + sigma4)*I4[-1])*dt
        next_H4 = H4[-1] + (rho4*I4[-1] - (eta + phi4)*H4[-1])*dt
        next_R4 = R4[-1] + (eta*H4[-1] + theta*I4[-1])*dt
        
        next_S5 = S5[-1] - (p*S5[-1] - gamma*R4[-1] - gamma*R5[-1] + epsilon*S5[-1])*dt
        next_E5 = E5[-1] + (p*S5[-1] - alpha*E5[-1])*dt
        next_I5 = I5[-1] + (alpha*E5[-1] - (theta + rho5 + sigma5)*I5[-1])*dt
        next_H5 = H5[-1] + (rho4*I5[-1] - (eta + phi5)*H5[-1])*dt
        next_R5 = R5[-1] + (eta*H5[-1] + theta*I5[-1] - gamma*R5[-1])*dt
        
        next_D  = D[-1] + (phi1*H1[-1] + sigma1*I1[-1] + phi2*H2[-1] + sigma2*I2[-1] + phi3*H3[-1] + sigma3*I3[-1]+ phi4*H4[-1] + sigma4*I4[-1] + phi5*H5[-1] + sigma5*I5[-1])*dt
        
        next_V  = V[-1] - (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*betav*V[-1] + mu*V[-1] - epsilon*(S1[-1]+S5[-1]) + pv*V[-1])*dt
        next_Ev = Ev[-1] + (c.dot((I1[-1] + I2[-1] + I3[-1] + I4[-1] + I5[-1] + Iv[-1])*N)*betav*V[-1] - alpha*Ev[-1] + pv*V[-1])*dt
        next_Iv = Iv[-1] + (alpha*Ev[-1] - theta*Iv[-1] - rhov*Iv[-1])*dt
        next_Hv = Hv[-1] + (rhov*Iv[-1] - eta*Hv[-1])*dt
        next_Rv = Rv[-1] + (eta*Hv[-1] + theta*Iv[-1] - gammav*Rv[-1])*dt
        
        
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
        Hv = np.vstack((Hv, next_Hv))
        Rv = np.vstack((Rv, next_Rv))
    
    return S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Hv, Rv, Ns, Nh


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


file   = pd.ExcelFile('RSV_24___44.xlsx')
# df     = file.parse(3) 
# idata  = df.values[0:, 1:8]
# hdata  = df.values[0:, 20:27]

df1    = file.parse(30)#30 #12 #17 #20 #23
idata  = df1.values[0:, 2:9]
df2    = file.parse(18) 
hdata  = df2.values[0:, 1:8]

it = 104 #104
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters		

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
mu      = 0#1./90
alpha   = 1./7
eta     = 1./5
phi     = 0#eta
epsilon = np.array([0.05, 0.05, 0, 0, 0, 0, 0])
# epsilon = np.array([1.35, 1.35, 0, 0, 0, 0.5, 0.5]) # monoclonal & vaccine

# period  = P1, P2, P3, P4
params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho

N = 83166711         # worldometer data

# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.2365, 0.3709, 0.2057, 0.0428])
N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.3709, 0.2057, 0.0428])

# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.03709, 0.02057, 0.00428]) #ARE, RKI & Muspad

Ns_0 = idata[0]/N
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


# #### init run 2021 RKI & MuSPAD
# init_vals = (np.array([1.45534666e-05, 1.60037366e-05, 5.98572249e-01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.61310233e-09, 1.01721254e-09, 8.91701798e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([7.10418407e-10, 4.82818328e-10, 7.20417522e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([6.34853680e-10, 2.16455136e-10, 1.97955575e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.52919015e-05, 4.99775715e-06, 3.90495681e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.66090538, 0.70355637, 0.20031736, 0.        , 0.        , 0.        , 0.        ]),
#  np.array([1.28755885e-05, 1.71104120e-05, 2.23696467e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([6.15600106e-06, 6.26977537e-06, 2.52898989e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([3.63352910e-06, 1.38132096e-06, 6.54526176e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.83657263e-03, 6.50670912e-04, 1.14848905e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.13621771, 0.11200039, 0.10007359, 0.        , 0.        , 0.        , 0.        ]),
#  np.array([1.77517317e-06, 1.82517662e-06, 7.45164513e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.16280396e-06, 8.02886850e-07, 1.41908642e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([5.63759728e-07, 1.50188326e-07, 3.21024719e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([2.74936161e-03, 8.78054896e-04, 4.76215056e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.10120947, 0.10031037, 0.10001519, 0.        , 0.        , 0.        , 0.        ]),
#  np.array([6.69239799e-07, 8.19107402e-07, 3.72423518e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([7.54640126e-07, 5.10272441e-07, 1.73284632e-08, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([2.57653780e-07, 7.56867469e-08, 2.59405488e-09, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([2.74936161e-03, 8.78054896e-04, 4.76215056e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([9.88325109e-03, 3.10963628e-03, 1.72683818e-04, 9.99451957e-01, 9.99318771e-01, 9.99046136e-01, 9.97383339e-01]),
#  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 3.24074805e-07, 3.07023398e-07, 1.02511415e-05, 3.47727485e-06]),
#  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 5.22219311e-06, 5.63944697e-06, 9.28331238e-06, 3.14346923e-05]),
#  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 6.69430692e-07, 6.05628114e-07, 7.81097077e-07, 3.06259343e-06]),
#  np.array([0.        , 0.        , 0.        , 0.00041594, 0.00046934, 0.0006274 , 0.00176382]),
#  np.array([0.02578501, 0.00844977, 0.00091853, 0.00060541, 0.00062892, 0.00093747, 0.00249495]),
#  np.array([0.07004251, 0.07370819, 0.        , 0.        , 0.        , 0.        , 0.        ]),
#  np.array([1.11385836e-06, 1.29094602e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([1.21832352e-05, 4.57290487e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
#  np.array([0.00087313, 0.00029674, 0.        , 0.        , 0.        , 0.        , 0.        ]),
#  np.array([0.08184637874068951, 0.026686680590009976, 0.001963659362269918, 0.006165203598712996, 0.006500782430498748, 0.009699830065996609, 0.025796133757925076], dtype=object),
#  np.array([0.03164966899408866, 0.0078009251883809725, 0.000532516785945772, 0.0003197666744036966, 0.00028266528073324506, 0.00042219721920451193, 0.0011262251603684442], dtype=object))


# init_vals[31] = np.zeros(7)

# opparams    = np.ones(7)*0.001
# opp         = np.ones((len(times), 7))*0.001

# init_vals   = S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
# # modelparams = init_vals, params, N, t
# S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]       

# for i, j in enumerate(times):
#     if i > 0:
#         dummyt = np.linspace(j, j+7, 8)
#         modelparams = init_vals, params, N, dummyt
        
#         optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, i), tol=1e-10, bounds=bounds)
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
# zNs = np.zeros((it,7))
# for i in range(it-1):
#     zNs[i] = zN[i+1]-zN[i]
    
# zR  = R5[::7]*N
# zS  = S5[::7]*N



# ####### PREDICTION
# S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Hv, Rv, Ns, Nh 

init = (np.array([0.01293421, 0.01309259, 0.59940977, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([2.80251920e-11, 9.29488902e-12, 4.33311817e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.27229789e-11, 6.05378553e-12, 3.57796631e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.14642268e-11, 3.20335373e-12, 9.21565488e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([4.32779397e-09, 1.41460447e-09, 1.23181352e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.76621818, 0.77971769, 0.19998014, 0.        , 0.        , 0.        , 0.        ]),
  np.array([1.16866389e-03, 3.84137521e-04, 1.08397164e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([7.04601731e-04, 3.24200381e-04, 1.13752080e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([5.38949138e-04, 1.44299323e-04, 2.49970810e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([4.81659626e-03, 1.58053166e-03, 3.44212106e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.10416198, 0.10151899, 0.09998661, 0.        , 0.        , 0.        , 0.        ]),
  np.array([1.04246690e-04, 3.32382996e-05, 3.61324561e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([9.33946850e-05, 4.00354291e-05, 5.02491555e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([5.52323517e-05, 1.36691936e-05, 8.40768361e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([4.88540751e-04, 1.55242970e-04, 1.29082620e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([0.09967332, 0.09989372, 0.09998897, 0.        , 0.        , 0.        , 0.        ]),
  np.array([5.05633542e-05, 1.64280446e-05, 1.80669403e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([7.82344215e-05, 3.10631027e-05, 3.50044372e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([2.62073643e-05, 6.02953898e-06, 3.34773962e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([6.00987362e-04, 2.05211562e-04, 1.51662476e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
  np.array([1.61205753e-03, 5.63857295e-04, 3.33534226e-05, 9.98816398e-01, 9.98729050e-01, 9.97003552e-01, 9.88564046e-01]),
  np.array([0.        , 0.        , 0.        , 0.00014622, 0.00015774, 0.00038674, 0.0015927 ]),
  np.array([0.        , 0.        , 0.        , 0.00033458, 0.00036702, 0.00089234, 0.00346277]),
  np.array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.54021653e-05, 2.29245505e-05, 5.53258446e-05, 2.10245118e-04]),
  np.array([0.        , 0.        , 0.        , 0.0006442 , 0.00066992, 0.00155895, 0.00580054]),
  np.array([0.00811548, 0.00278633, 0.00032067, 0.0001684 , 0.0001676 , 0.00032583, 0.00117045]),
  np.array([0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]),
  np.array([1.56458194e-05, 5.12820011e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]), #Ev
  np.array([5.34623017e-05, 1.76338877e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]), #Iv
  np.array([0.        , 0.        , 0.        , 0. , 0., 0., 0.]),  #Hv
  np.array([1.01500570e-04, 3.38303437e-05, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]), #Rv
  np.array([0.        , 0.        , 0.        , 0. , 0., 0., 0.]), #Ns
  np.array([0.        , 0.        , 0.        , 0. , 0., 0., 0.])) #Nh
  # np.array([0.02369419453452381, 0.00815403117638834, 0.0008115327688335656, 0.002222281841834497, 0.0022767501269243284, 0.004694456811321734, 0.017333718799779074]),
  # np.array([0.010261184444026392, 0.002634343127172547, 0.00018937424870404222, 9.021014231147958e-05, 7.645433002253255e-05, 0.0001498093389235152, 0.0005400086889255369]))


df3  = file.parse(41)  
sea  = df3.values[157:, 1:8] #104 German, 156 Saxon, 157 ARE
sea  = df3.values[0:, 26:33]

df6  = file.parse(40) 
ser  = df6.values[52:, 20:27] #RKI

ft  = np.arange(0, 52*7, 7)

# init_vals   = S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
# init_vals = fS1[-1], fE1[-1], fI1[-1], fH1[-1], fR1[-1], fS2[-1], fE2[-1], fI2[-1], fH2[-1], fR2[-1], fS3[-1], fE3[-1], fI3[-1], fH3[-1], fR3[-1], fS4[-1], fE4[-1], fI4[-1], fH4[-1], fR4[-1], fS5[-1], fE5[-1], fI5[-1], fH5[-1], fR5[-1], fD[-1], fV[-1], fEv[-1], fIv[-1], fRv[-1], fNs[-1], fNh[-1]
init_vals = init

gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gHv, gRv, gNs, gNh = init_vals   
    
for i, j in enumerate(ft):
    
    par = np.array([0.4, 0.5, 1, 1.5, 1.5, 1, 1]) # ARE
    # par = np.array([0.15142, 0.06557, 0.00562, 0.0005, 0.000030, 0.00004, 0.00008]) #* 1.2 #german
    # par = np.array([0.15142, 0.06557, 0.00562, 0.2075/30, 0.0890/30, 0.0889/30, 0.0833/30]) * 1 # Muspad
    # par = np.array([0.70947/3.7, 0.3358/2.7, 0.01652/1.5, 0.0032/2, 0.00020/2.5, 0.00036/4, 0.00058/4]) * 1. #saxony
    # par = np.array([0.70947/3.7, 0.3358/2.7, 0.01652/1.5, 0.2075/10, 0.0890/10, 0.0889/10, 0.0833/10]) * 1. #saxony muspad
    # par = np.array([0.70947, 0.33580, 0.01652, 0.2264/15, 0.1402/15, 0.2222/15, 0.2222/0.1])
    # par = np.array([0.70947/2, 0.33580/2, 0.01652/2, 0.0941/20, 0.0576/20, 0.0537/20, 0.0326/20]) 
    
    
    # par = par*0.7  #low
    # par = par*1.3  #up
    
    par = sea[i] * par
    
    # r = np.array([0.0052341, 0.0359557, 0.0052, 0.1199, 0.01704, 0.01506, 0.001303])* 0.65 #german saxony
    # r = np.array([0.0052341, 0.0359557, 0.0052, 0.031869, 0.001143, 0.004506, 0.001303])*0.3  # Muspad
    r = np.array([0.0052341, 0.0359557, 0.0052, 0.1199, 0.01704, 0.01506, 0.001303])*0.1
    # r = np.array([0.43341, 0.69557, 0.029917, 0.031869, 0.001143, 0.011624, 0.010319]) 
    
    # r = r*0.7  #low
    r = r*1.3  #up
    
    rho = ser[i] * r
    
    # params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho
    
    dummyt = np.linspace(0, 7, 8)
    nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nHv, nRv, nNs, nNh = model(init_vals, params, par, dummyt, rho)
    init_vals = nS1[-1], nE1[-1], nI1[-1], nH1[-1], nR1[-1], nS2[-1], nE2[-1], nI2[-1], nH2[-1], nR2[-1], nS3[-1], nE3[-1], nI3[-1], nH3[-1], nR3[-1], nS4[-1], nE4[-1], nI4[-1], nH4[-1], nR4[-1], nS5[-1], nE5[-1], nI5[-1], nH5[-1], nR5[-1], nD[-1], nV[-1], nEv[-1], nIv[-1], nHv[-1], nRv[-1], nNs[-1], nNh[-1]
        
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
    gHv = np.vstack((gHv, nHv[1:,:]))
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



# num = 10
# simulations = np.zeros((num, 52*7+1, 7))
# quantiles = [0.025, 0.975]

# for i in range(num):
#     gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gRv, gNs, gNh = init   
#     init_vals   = init # S_0, E_0, I_0, H_0, R_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0
    
#     opp = np.array([0.15142, 0.06557, 0.00562, 0.0005, 0.000030, 0.00004, 0.00008])
#     opp = np.random.normal(opp, opp*2, 7)
#     opp[opp<0] = 0
    
#     for j in range(52):
#         dummyt = np.linspace(0, 7, 8)
        
#         opp = opp * sea[j] 
        
#         r = np.array([0.0052341, 0.0359557, 0.0052, 0.1199, 0.01704, 0.01506, 0.001303]) * 1 
    
#         rho = ser[j] * r

            
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

# # m_result = np.mean(simulations, axis=0)
# qr = np.zeros((2, 52*7+1, 7))
# for i in range(2):
#     qr[i] = np.percentile(simulations, quantiles[i] * 100, axis=0)

# zQ = qr[:,::7]
# zNq = np.zeros((2,52,7))
# for i in range(51):
#     zNq[:,i+1] = zQ[:,i+1]-zQ[:,i]


# # zi = np.zeros((105,7))
# # for i in range(105):
# #     zi[i] = sum(idata[:i+1])

# color = ['purple', 'orange', 'green', 'cyan', 'blue', 'grey', 'red']
# label = ['0-1 years', '2-4 years', '5-14 years', '15-34 years', '35-59 years', '60-79 years', '80+ years']

color = ['purple', 'red', 'green', 'blue', 'orange', 'grey', 'yellow']
label = ['A0-1', 'A2-4', 'A5-14', 'A15-34', 'A35-59', 'A60-79', 'A80+']

# for i in range(7):
#     # plt.fill_between(ft[:-1]/7, zNq[1,:-1,i], zNq[0,:-1,i], color=color[i], alpha=0.2)
# plt.xlabel('Week')
# plt.ylabel('New cases')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# # plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.show()

# # ##### No intervantion
zN2h[:24,0] = zN2h[:24,0]*20
zN2h[24:,0] = zN2h[24:,0]*50
zN2h[:,1:] = zN2h[:,1:]*50

#### Vaccine
# zN2h[:24,0] = zN2h[:24,0]*10
# zN2h[24:,0] = zN2h[24:,0]*25
# zN2h[:,1:] = zN2h[:,1:]*30


# for i in range(7):
#     plt.plot(ft[:-1]/7,zN2s[:-1,i], color=color[i], label=label[i], linestyle='solid') 
#     # plt.plot(ft[:-1]/7,zN2h[:-1,i], color=color[i], label=label[i], linestyle='dashdot') 
#     # # plt.fill_between(ft[:-1]/7, zQ[1,:-1,i], zQ[0,:-1,i], color=color[i], alpha=0.2)
# plt.xlabel('Week')
# plt.ylabel('New cases')
# # plt.ylim(0,2300)
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.grid()
# plt.show()

# for i in range(7):
#     plt.plot(ft[:-1]/7,zN2h[:-1,i], color=color[i], label=label[i]) 
# plt.xlabel('Week')
# plt.ylabel('New hospitalisation')
# # plt.ylim(0,1850)
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.show()
    
file   = pd.ExcelFile('RSV_202425newsc.xlsx')
# 43 - 51   ## Vaccine 52 - 53 # 47 scenario0

# MusPAD 54, 55, 56
# RKI 47, 52, 53
df1    = file.parse(56)
ri  = df1.values[56:108, 2:9]  #1:
li  = df1.values[56:108, 10:17]
ui  = df1.values[56:108, 18:25]
rh  = df1.values[56:108, 26:33]
lh  = df1.values[56:108, 34:41]
uh  = df1.values[56:108, 42:49]

ri  = ri.astype(float)
li  = li.astype(float)
ui  = ui.astype(float)
rh  = rh.astype(float)
lh  = lh.astype(float)
uh  = uh.astype(float)

file1 = pd.ExcelFile('RSVpaper_2.xlsx')
df2   = file1.parse(4) 
hdata = df2.values[365:418, 2:9]
df3   = file1.parse(5) 
idata = df3.values[365:418, 2:9]

# for i in range(7):
#     plt.plot(ft[:-1]/7,ri[:-1,i], color=color[i], label=label[i], linestyle='solid') 
#     plt.plot(ft[:-1]/7,rh[:-1,i], color=color[i], label=label[i], linestyle='dashdot') 
# plt.xlabel('Week')
# # plt.ylabel('New cases')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.show()


# for i in range(7):
#     plt.plot(ft[:-1]/7,ri[:-1,i], color=color[i], label=label[i]) 
#     # plt.plot(ft[:-1]/7,ui[:-1,i], color=color[i]) 
#     # plt.plot(ft[:-1]/7,li[:-1,i], color=color[i]) 
#     plt.fill_between(ft[:-1]/7, ui[:-1,i], li[:-1,i], color=color[i], alpha=0.2)
# plt.xlabel('Week')
# plt.ylabel('New cases')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.ylim(0,66000)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.show()

# for i in range(7):
#     plt.plot(ft[:-1]/7,rh[:-1,i], color=color[i], label=label[i]) 
#     # plt.plot(ft[:-1]/7,uh[:-1,i], color=color[i]) 
#     # plt.plot(ft[:-1]/7,lh[:-1,i], color=color[i]) 
#     # plt.fill_between(ft[:-1]/7, uh[:-1,i]*1.4, lh[:-1,i]*0.7, color=color[i], alpha=0.2)
#     plt.fill_between(ft[:-1]/7, uh[:-1,i], lh[:-1,i], color=color[i], alpha=0.2)
# plt.xlabel('Week')
# plt.ylabel('New hospitalisation')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.ylim(0,2100)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.show()

#### RKI
# fig = plt.figure(figsize=(6, 4), dpi=300)
# ax  = fig.add_subplot(1,1,1)
# for i in range(7):
#     plt.plot(idata[:,i], color=color[i], marker='o', markersize=4, linewidth=0) 
#     plt.plot(ft[:-1]/7,ri[:-1,i], color=color[i], label=label[i]) 
#     # plt.plot(ft[:-1]/7,ui[:-1,i], color=color[i]) 
#     # plt.plot(ft[:-1]/7,li[:-1,i], color=color[i]) 
#     plt.fill_between(ft[:-1]/7, ui[:-1,i], li[:-1,i], color=color[i], alpha=0.2)
# plt.xlabel('Week')
# plt.ylabel('New cases')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.ylim(0,11000)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')

# styles = [' ', '-']
# mark   = ['o', ' ']
# labell = ['Notification data', 'Model prediction']
# ax2 = ax.twinx()
# for ss, sty in enumerate(styles):
#     ax2.plot(np.NaN, np.NaN, ls=styles[ss], marker=mark[ss], markersize=5, label=labell[ss], c='black')
# ax2.get_yaxis().set_visible(False)
# ax2.legend(bbox_to_anchor=(0.5, -0.21), borderaxespad=0, ncol=2, loc="center",fontsize='10')
# plt.show()

# fig = plt.figure(figsize=(6, 4), dpi=300)
# ax  = fig.add_subplot(1,1,1)
# for i in range(7):
#     plt.plot(hdata[:,i], color=color[i], marker='o', markersize=4, linewidth=0) 
#     plt.plot(ft[:-1]/7,rh[:-1,i], color=color[i], label=label[i]) 
#     # plt.plot(ft[:-1]/7,uh[:-1,i], color=color[i]) 
#     # plt.plot(ft[:-1]/7,lh[:-1,i], color=color[i]) 
#     # plt.fill_between(ft[:-1]/7, uh[:-1,i]*1.4, lh[:-1,i]*0.7, color=color[i], alpha=0.2)
#     plt.fill_between(ft[:-1]/7, uh[:-1,i], lh[:-1,i], color=color[i], alpha=0.2)
# plt.xlabel('Week')
# plt.ylabel('New hospitalisation')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.ylim(0,8500)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')

# styles = [' ', '-']
# mark   = ['o', ' ']
# labell = ['Notification data', 'Model prediction']
# ax2 = ax.twinx()
# for ss, sty in enumerate(styles):
#     ax2.plot(np.NaN, np.NaN, ls=styles[ss], marker=mark[ss], markersize=5, label=labell[ss], c='black')
# ax2.get_yaxis().set_visible(False)
# ax2.legend(bbox_to_anchor=(0.5, -0.21), borderaxespad=0, ncol=2, loc="center",fontsize='10')
# plt.show()

#### MuSPAD
plt.figure(figsize=(6, 4),dpi=300)
for i in range(7):
    plt.plot(ft[:-1]/7,ri[:-1,i], color=color[i], label=label[i]) 
    # plt.plot(ft[:-1]/7,ui[:-1,i], color=color[i]) 
    # plt.plot(ft[:-1]/7,li[:-1,i], color=color[i]) 
    plt.fill_between(ft[:-1]/7, ui[:-1,i], li[:-1,i], color=color[i], alpha=0.2)
plt.xlabel('Week')
plt.ylabel('New cases')
plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
plt.ylim(0,220000) #(0,3700)
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
plt.show()

fig = plt.figure(figsize=(6, 4), dpi=300)
ax  = fig.add_subplot(1,1,1)
for i in range(7):
    plt.plot(hdata[:,i], color=color[i], marker='o', markersize=4, linewidth=0) 
    plt.plot(ft[:-1]/7,rh[:-1,i], color=color[i], label=label[i]) 
    # plt.plot(ft[:-1]/7,uh[:-1,i], color=color[i]) 
    # plt.plot(ft[:-1]/7,lh[:-1,i], color=color[i]) 
    # plt.fill_between(ft[:-1]/7, uh[:-1,i]*1.4, lh[:-1,i]*0.7, color=color[i], alpha=0.2)
    plt.fill_between(ft[:-1]/7, uh[:-1,i], lh[:-1,i], color=color[i], alpha=0.2)
plt.xlabel('Week')
plt.ylabel('New hospitalisation')
plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
plt.ylim(0,4400) #(0,3700)
plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')

styles = [' ', '-']
mark   = ['o', ' ']
labell = ['Notification data', 'Model prediction']
ax2 = ax.twinx()
for ss, sty in enumerate(styles):
    ax2.plot(np.NaN, np.NaN, ls=styles[ss], marker=mark[ss], markersize=5, label=labell[ss], c='black')
ax2.get_yaxis().set_visible(False)
ax2.legend(bbox_to_anchor=(0.5, -0.21), borderaxespad=0, ncol=2, loc="center",fontsize='10')
plt.show()

# # 1-43 2-44 3-45 4-46 5-47 6-48 7-49 8-50 9-51
# df1  = file.parse(46)
# val1 = df1.values[1:, 1:]
# val1 = val1.astype(float)

# df2  = file.parse(47)
# val2 = df2.values[1:, 1:]
# val2 = val2.astype(float)

# df3  = file.parse(48)
# val3 = df3.values[1:, 1:]
# val3 = val3.astype(float)

# # 8 INFECTION
# # 32 HOSPITAL
# # plt.plot(ft[:-1]/7,val1[:-1,8], color='blue', label='Scenario 2') 
# # plt.plot(ft[:-1]/7,val2[:-1,8], color='orange', label='Scenario 5') 
# # plt.plot(ft[:-1]/7,val3[:-1,8], color='red', label='Scenario 8') 
# # plt.xlabel('Week')
# # plt.ylabel('New cases')
# plt.plot(ft[:-1]/7,val1[:-1,32]*50, color='blue', label='Scenario 4') 
# plt.plot(ft[:-1]/7,val2[:-1,32]*50, color='orange', label='Scenario 5') 
# plt.plot(ft[:-1]/7,val3[:-1,32]*50, color='red', label='Scenario 6') 
# plt.xlabel('Week')
# plt.ylabel('New hospitalisations')
# plt.xticks(np.arange(0, 51, 4), np.hstack((np.arange(21,52,4), np.arange(1,20,4))), rotation=0)
# plt.legend(bbox_to_anchor=(1.04,1), borderaxespad=0, loc="upper left",fontsize='medium')
# plt.show()




# zN  = fNs[::7]*N
# zNs = np.zeros((5,6))
# for i in range(5):
#     zNs[i] = zN[i+1]-zN[i]
    
# for i in range(6):
#     plt.plot(np.linspace(1, 47, 47), idata[:,i], color=color[i])
#     # plt.plot(np.linspace(47, 48), [idata[-1,i], zNs[0,i]], color=color[i])
#     plt.plot(np.linspace(48, 53, 5),zNs[:,i], color=color[i])

# # import csv
# # with open('file.csv', 'w', newline='') as f:
# #     writer = csv.writer(f)
# #     writer.writerows(init_vals)
# df     = file.parse(42) 
# idata  = df.values[0:365, 2:9]
# fig = plt.figure( figsize=(20, 5))
# for i in range(7):
#     plt.plot(np.linspace(1, 364, 364),idata[:-1,i], color=color[i], label=label[i]) 
# plt.xlabel('Week')
# plt.ylabel('New hospitalisations')
# plt.legend(bbox_to_anchor=(0.5, -0.3), borderaxespad=0, ncol=7, loc="center",fontsize='10')
# plt.xticks(np.arange(1, 364, 4), np.hstack((np.arange(21,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,20,4))), rotation=0)
# plt.axvline(x=53, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=105, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=157, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=210, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=262, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=315, linewidth = 1, color='black', linestyle='dashed')
# plt.text(20, -1850, '17/18', fontsize=10)
# plt.text(72, -1850, '18/19', fontsize=10)
# plt.text(124, -1850, '19/20', fontsize=10)
# plt.text(176, -1850, '20/21', fontsize=10)
# plt.text(228, -1850, '21/22', fontsize=10)
# plt.text(280, -1850, '22/23', fontsize=10)
# plt.text(332, -1850, '23/24', fontsize=10)

# file   = pd.ExcelFile('RSV1.xlsx')
# df     = file.parse(9) 
# idata  = df.values[0:365, 2:9]
# fig = plt.figure( figsize=(20, 5))
# for i in range(7):
#     plt.plot(np.linspace(1, 364, 364),idata[:-1,i], color=color[i], label=label[i]) 
# plt.xlabel('Week')
# plt.ylabel('New cases')
# plt.legend(bbox_to_anchor=(0.5, -0.3), borderaxespad=0, ncol=7, loc="center",fontsize='10')
# plt.xticks(np.arange(1, 364, 4), np.hstack((np.arange(21,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,52,4), np.arange(1,20,4))), rotation=0)
# plt.axvline(x=53, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=105, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=157, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=210, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=262, linewidth = 1, color='black', linestyle='dashed')
# plt.axvline(x=315, linewidth = 1, color='black', linestyle='dashed')
# plt.text(20, -500, '17/18', fontsize=10)
# plt.text(72, -500, '18/19', fontsize=10)
# plt.text(124, -500, '19/20', fontsize=10)
# plt.text(176, -500, '20/21', fontsize=10)
# plt.text(228, -500, '21/22', fontsize=10)
# plt.text(280, -500, '22/23', fontsize=10)
# plt.text(332, -500, '23/24', fontsize=10)