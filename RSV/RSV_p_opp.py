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
# bounds = Bounds(np.zeros((14)), 100*np.ones((14)))


# def model(init_vals, params, opparams, t, par):
def model(init_vals, params, par, dummyt):
    S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0 = init_vals
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]
    
    c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho = params
    
    # c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi = params
    beta     = np.zeros(7)
    p        = np.zeros(7)
    beta[:3] = par[:3]
    p[3:]    = par[3:7]
    rho      = opparams
    
    # # c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi = params
    # beta     = np.zeros(7)
    # p        = np.zeros(7)
    # beta[:3] = opparams[:3]
    # p[3:]    = opparams[3:7]
    # rho      = opparams[7:]
    
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
        
        # next_Nh  = next_H1 + next_H2 + next_H3 + next_H4 + next_H5 
        
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


def cost(opparams, modelparams, idata, hdata, i, par):
    init_vals, params, N, t = modelparams
    S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = model(init_vals, params, opparams, t, par)
        
    dum = 0
    
    # dui  = sum(idata[:i+1])
    # simi = Ns[7,:] * N
    # if max(dui) == 0:
    #     p = (simi - dui)**2
    # else:
    #     p = (simi - dui)**2 * 1./max(dui)
    # dum = dum + sum(p)
    
    duh  = sum(hdata[:i+1])
    simh = Nh[7,:] * N
    q    = (simh - duh)**2 * 1./max(duh)
    dum = dum + sum(q)

    return dum


file   = pd.ExcelFile('RSV_202425.xlsx')

df1    = file.parse(30) # based on data source
idata  = df1.values[209:, 2:9]
df2    = file.parse(34) 
hdata  = df2.values[209:, 2:9]

it = 155#313 #104
t_max = 7*it
times = np.arange(0, t_max+1, 7)

dt = 1
t  = np.linspace(0, t_max, int(t_max/dt)+1)

### Parameters		
##polymod
# c   = 10 * np.array([[3.60E-07, 3.00E-07, 1.03E-07, 2.32E-07, 6.40E-08, 3.23E-08, 0.00E+00],
#                       [3.00E-07, 9.19E-07, 1.09E-07, 8.59E-08, 6.49E-08, 4.01E-08, 2.25E-08],
#                       [1.03E-07, 1.09E-07, 4.22E-07, 7.46E-08, 7.68E-08, 3.21E-08, 1.37E-08],
#                       [2.32E-07, 8.59E-08, 7.46E-08, 2.60E-07, 1.03E-07, 3.77E-08, 2.61E-08],
#                       [6.40E-08, 6.49E-08, 7.68E-08, 1.03E-07, 1.31E-07, 5.91E-08, 3.37E-08],
#                       [3.23E-08, 4.01E-08, 3.21E-08, 3.77E-08, 5.91E-08, 1.19E-07, 6.09E-08],
#                       [0.00E+00, 2.25E-08, 1.37E-08, 2.61E-08, 3.37E-08, 6.09E-08, 8.79E-08]])


# c  = 10 * np.array([[2.60E-07, 2.00E-07, 9.03E-08, 1.32E-07, 5.40E-08, 2.23E-08, 0.00E+00],
#                      [2.00E-07, 9.19E-07, 1.09E-07, 8.59E-08, 6.49E-08, 4.01E-08, 2.25E-08],
#                      [9.03E-08, 1.09E-07, 4.22E-07, 7.46E-08, 7.68E-08, 3.21E-08, 1.37E-08],
#                      [1.32E-07, 8.59E-08, 7.46E-08, 2.60E-07, 1.03E-07, 3.77E-08, 2.61E-08],
#                      [5.40E-08, 6.49E-08, 7.68E-08, 1.03E-07, 1.31E-07, 5.91E-08, 3.37E-08],
#                      [2.23E-08, 4.01E-08, 3.21E-08, 3.77E-08, 5.91E-08, 1.19E-07, 6.09E-08],
#                      [0.00E+00, 2.25E-08, 1.37E-08, 2.61E-08, 3.37E-08, 6.09E-08, 8.79E-08]])

##covimod
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
params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi
# params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi

N = 83166711         # worldometer data

# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.2365, 0.3709, 0.2057, 0.0428])
# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.002365, 0.003709, 0.002057, 0.000428]) #RKI & Muspad
# N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.03709, 0.02057, 0.00428]) #ARE
N    = N * np.array([0.0176, 0.0264, 0.1001, 0.02365, 0.3709, 0.2057, 0.0428])
Ns_0 = idata[0]/N
Nh_0 = hdata[0]/N #np.zeros(7)

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
# df            = file.parse(13)
# init_vals     = df.values[0:32, 10:17]
# idata         = idata[209:]
# hdata         = hdata[209:]

S1_0, E1_0, I1_0, H1_0, R1_0, S2_0, E2_0, I2_0, H2_0, R2_0, S3_0, E3_0, I3_0, H3_0, R3_0, S4_0, E4_0, I4_0, H4_0, R4_0, S5_0, E5_0, I5_0, H5_0, R5_0, D_0, V_0, Ev_0, Iv_0, Rv_0, Ns_0, Nh_0 = init_vals    
# Ns_0 = idata[0]/N
# Nh_0 = hdata[0]/N#np.zeros(7)#hdata[0]/N
####################################

opparams    = np.ones(7)*0.001
opp         = np.ones((len(times), 7))*0.001

# df4 = file.parse(26) #33#26
# b   = df4.values[0:, 53:60]
# b   = df4.values[0:, 41:48]

df3  = file.parse(29)#6 #12 #17 #20 #23
sea  = df3.values[0:, 1:8]

# opparams    = np.ones(14)*0.001
# opp         = np.ones((len(times), 14))*0.001

S1, E1, I1, H1, R1, S2, E2, I2, H2, R2, S3, E3, I3, H3, R3, S4, E4, I4, H4, R4, S5, E5, I5, H5, R5, D, V, Ev, Iv, Rv, Ns, Nh = [S1_0], [E1_0], [I1_0], [H1_0], [R1_0], [S2_0], [E2_0], [I2_0], [H2_0], [R2_0], [S3_0], [E3_0], [I3_0], [H3_0], [R3_0], [S4_0], [E4_0], [I4_0], [H4_0], [R4_0], [S5_0], [E5_0], [I5_0], [H5_0], [R5_0], [D_0], [V_0], [Ev_0], [Iv_0], [Rv_0], [Ns_0], [Nh_0]       

# for i, j in enumerate(times):
#     # print(i)
#     if i > 0:
#         dummyt = np.linspace(0, 7, 8)

        
#         par = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058])
        
#         # par = np.array([0.70947, 0.33580, 0.01652, 0.0941/30, 0.0576/30, 0.0537/30, 0.0326/45]) #Muspad
#         # if i >= 52:
#         #     par = np.array([0.70947, 0.33580, 0.01652, 0.2075/60, 0.0976/60, 0.1360/60, 0.1667/90])
#         # if i >= 104:
#         #     par = np.array([0.70947, 0.33580, 0.01652, 0.0/60, 0.1007/60, 0.0889/60, 0.0833/90])
            
#         par = sea[i] * par
        
#         # params  = c1, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, par
        
#         # if i == 156:
#             # params  = c1, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho
#             # params  = c1, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, b[i-1]
#             # params  = c1, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi

#         modelparams = init_vals, params, N, dummyt
        
#         optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, hdata, i, par), tol=1e-10, bounds=bounds)
#         # optimizer = opt.minimize(cost, opparams, args=(modelparams, idata, hdata, i), tol=1e-10)
#         opp[i] = optimizer.x
    
#         nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, opp[i], dummyt, par) 
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
# for i in range(it - 1):
#     zNs[i+1] = zN[i+1]-zN[i]
# zNs[0] = idata[0]

# zH  = Nh[::7]*N
# zNh = np.zeros((it,7))
# for i in range(it - 1):
#     zNh[i+1] = zH[i+1]-zH[i]
# zNh[0] = hdata[0]
    
# zR  = R5[::7]*N
# zS  = S5[::7]*N

df3  = file.parse(17)#6 #12 #17 #20 #23
sea  = df3.values[0:, 1:8]

ft  = np.arange(0, 104*7, 7)
# S1[1463], E1[1463], I1[1463], H1[1463], R1[1463], S2[1463], E2[1463], I2[1463], H2[1463], R2[1463], S3[1463], E3[1463], I3[1463], H3[1463], R3[1463], S4[1463], E4[1463], I4[1463], H4[1463], R4[1463], S5[1463], E5[1463], I5[1463], H5[1463], R5[1463], D[1463], V[1463], Ev[1463], Iv[1463], Rv[1463], Ns[1463], Nh[1463]

df4  = file.parse(11)#6 #12 #17 #20 #23
pr   = df4.values[209:, 13:20]
params  = c, sea, sigma, theta, gamma, gammav, mu, epsilon, alpha, eta, phi, rho

init_vals = S1[-1], E1[-1], I1[-1], H1[-1], R1[-1], S2[-1], E2[-1], I2[-1], H2[-1], R2[-1], S3[-1], E3[-1], I3[-1], H3[-1], R3[-1], S4[-1], E4[-1], I4[-1], H4[-1], R4[-1], S5[-1], E5[-1], I5[-1], H5[-1], R5[-1], D[-1], V[-1], Ev[-1], Iv[-1], Rv[-1], Ns[-1], Nh[-1]
fS1, fE1, fI1, fH1, fR1, fS2, fE2, fI2, fH2, fR2, fS3, fE3, fI3, fH3, fR3, fS4, fE4, fI4, fH4, fR4, fS5, fE5, fI5, fH5, fR5, fD, fV, fEv, fIv, fRv, fNs, fNh = init_vals   
    
for i, j in enumerate(ft):
    # par = np.array([0.47964, 0.24703, 0.01131, 0.00032, 0.00020, 0.00037, 0.00059])
    par = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058*1000]) #RKI
    # # par = np.array([0.70947, 0.33580, 0.01652, 0.0703/15, 0.0836/15, 0.0466/15, 0.0517/0.05]) #Muspad
    # # if i >= 357:
    # #     par = np.array([0.47964, 0.24703, 0.01131, 0.2264/15, 0.1402/15, 0.2222/15, 0.2222/0.05])
    # par = sea[i] * par
    
    par = pr[i]
    
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
   
# zN  = fNs[::7]*N
# zNs = np.zeros((it,7))
# for i in range(it - 1):
#     zNs[i+1] = zN[i+1]-zN[i]
# zNs[0] = idata[0]

# zH  = fNh[::7]*N
# zNh = np.zeros((it,7))
# for i in range(it - 1):
#     zNh[i+1] = zH[i+1]-zH[i]
# zNh[0] = hdata[0]
    
# zR  = fR5[::7]*N
# zS  = fS5[::7]*N

# ####### PREDICTION

df3  = file.parse(32)
sea  = df3.values[0:, 1:8]

ft  = np.arange(0, 104*7, 7)
# sea = np.ones(7) 

init_vals = fS1[-1], fE1[-1], fI1[-1], fH1[-1], fR1[-1], fS2[-1], fE2[-1], fI2[-1], fH2[-1], fR2[-1], fS3[-1], fE3[-1], fI3[-1], fH3[-1], fR3[-1], fS4[-1], fE4[-1], fI4[-1], fH4[-1], fR4[-1], fS5[-1], fE5[-1], fI5[-1], fH5[-1], fR5[-1], fD[-1], fV[-1], fEv[-1], fIv[-1], fRv[-1], fNs[-1], fNh[-1]
gS1, gE1, gI1, gH1, gR1, gS2, gE2, gI2, gH2, gR2, gS3, gE3, gI3, gH3, gR3, gS4, gE4, gI4, gH4, gR4, gS5, gE5, gI5, gH5, gR5, gD, gV, gEv, gIv, gRv, gNs, gNh = init_vals   
    
for i, j in enumerate(ft):
    if i <52:
        par = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058*800]) #Muspad
        # par = np.array([0.70947, 0.33580, 0.01652, 0.2264/15, 0.1402/15, 0.2222/15, 0.2222/0.1]) #RKI
        par = sea[i] * par
    else:
        par = np.array([0.70947, 0.33580, 0.01652, 0.00032, 0.00020, 0.00036, 0.00058*800]) #Muspad
        # par = np.array([0.70947, 0.33580, 0.01652, 0.2264/15, 0.1402/15, 0.2222/15, 0.2222/0.1]) #RKI
        par = sea[i-52] * par
    
    dummyt = np.linspace(0, 7, 8)
    nS1, nE1, nI1, nH1, nR1, nS2, nE2, nI2, nH2, nR2, nS3, nE3, nI3, nH3, nR3, nS4, nE4, nI4, nH4, nR4, nS5, nE5, nI5, nH5, nR5, nD, nV, nEv, nIv, nRv, nNs, nNh = model(init_vals, params, par, dummyt)
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
zN2s = np.zeros((104,7))
for i in range(103):
    zN2s[i] = zN2[i+1]-zN2[i]
    
zH2  = gNh[::7]*N
zN2h = np.zeros((104,7))
for i in range(103):
    zN2h[i] = zH2[i+1]-zH2[i]


# # zi = np.zeros((105,7))
# # for i in range(105):
# #     zi[i] = sum(idata[:i+1])

# color = ['purple', 'orange', 'green', 'cyan', 'blue', 'grey', 'red']
# label = ['0-1 years', '2-4 years', '5-14 years', '15-34 years', '35-59 years', '60-79 years', '80+ years']

# for i in range(7):
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

# # # import csv
# # # with open('file.csv', 'w', newline='') as f:
# # #     writer = csv.writer(f)
# # #     writer.writerows(init_vals)