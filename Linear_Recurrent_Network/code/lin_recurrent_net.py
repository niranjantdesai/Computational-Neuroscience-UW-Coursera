# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:36:17 2017

Calculate steady state output of a linear recurrent network

@author: Niranjan Thakurdesai
"""

import numpy as np
from numpy import linalg as LA

W = np.matrix('0.6 0.1 0.1 0.1 0.1; 0.1 0.6 0.1 0.1 0.1; 0.1 0.1 0.6 0.1 0.1;\
              0.1 0.1 0.1 0.6 0.1; 0.1 0.1 0.1 0.1 0.6')    # weight matrix
print("W = %s" %(W))
u = np.matrix('0.6; 0.5; 0.6; 0.2; 0.1')  # static input vector
print("u = %s" %(u))
M = np.matrix('-0.75 0 0.75 0.75 0; 0 -0.75 0 0.75 0.75; 0.75 0 -0.75 0 0.75;\
              0.75 0.75 0 -0.75 0; 0 0.75 0.75 0 -0.75')    # recurrent weight matrix
print("M = %s" %(M))

h = W*u
ev, e = LA.eig(M)

# Calculate coefficients of steady state output vector
# v = sum_i(c_i*e_i)
c = np.empty([ev.size])
v_ss = np.zeros([ev.size,1])
for i in range(ev.size):
    c[i] = h.T*e[:,i]/(1 - ev[i])
    v_ss += c[i]*e[:,i]
print("v_ss = %s" %(v_ss))