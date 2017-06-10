# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 00:01:19 2017

Implement a neuron that learns from two dimensional input data using Oja's Hebb rule

@author: Niranjan Thakurdesai
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pickle
with open('../data/c10p1.pickle', 'rb') as f:
    data = pickle.load(f)
u_mat = data['c10p1']

# Zero-mean centering
# wi+1=wi+Δtη(vu−αv^2w)
mu = np.mean(u_mat, axis=0)
num_points = u_mat.shape[0]
u_centered = u_mat - np.tile(mu, (num_points, 1))

plt.scatter(u_centered[:,0], u_centered[:,1])
plt.show()

# Implement learning rule
n = 1   # n = 1/time constant
alpha = 1
timestep = 0.01
w0 = np.random.rand(2)    # initial weight vector
max_iters = 100000
w = w0
iters = 0
for i in range(max_iters):
    u = u_centered[np.remainder(i,num_points),:]    # current input vector
    v = np.dot(u,w)    # current output
    w = w + timestep*n*(v*u - alpha*v*v*w)    # update

C = np.dot(u_centered.T, u_centered)/num_points   # input covariance matrix    
ev, e = LA.eig(C)
"""
The correlation matrix has only one principal eigenvector, but there are 
two vectors of length 1/√α that are parallel to this eigenvector. 
w can converge to either of these two vectors.
"""

"""
What happens when the data is not zero-mean centered before the learning process?
"""
# Add a constant offset to input data
offset = [5,10]
u_mat1 = u_mat + np.tile(offset, (num_points, 1))
mu1 = mu + offset
w1 = w0
iters = 0
for i in range(max_iters):
    u = u_mat1[np.remainder(i,num_points),:]    # current input vector
    v = np.dot(u,w1)    # current output
    w1 = w1 + timestep*n*(v*u - alpha*v*v*w1)    # update
"""
The two vectors that w converges to in different runs of the algorithm are parallel to the vector that points roughly towards the mean of the data.
"""
    
"""
What happens when the pure Hebb rule is used instead of Oja's rule?
"""
w2 = w0
iters = 0
for i in range(max_iters):
    u = u_centered[np.remainder(i,num_points),:]    # current input vector
    v = np.dot(u,w2)    # current output
    w2 = w2 + timestep*n*(v*u)    # update
"""
The vectors found by the learning rule have the same direction as those found by Oja's rule, but the length grows without bound as a function of the number of iterations.
"""