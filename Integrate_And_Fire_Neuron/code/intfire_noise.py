from __future__ import print_function
"""
Created on Wed Apr 22 16:02:53 2015

Basic integrate-and-fire neuron 
R Rao 2007

translated to Python by rkp 2015

Edited by Niranjan Thakurdesai 2017
"""

import numpy as np
import matplotlib.pyplot as plt


# input current
I0 = 1 # nA

# capacitance and leak resistance
C = 1 # nF
R = 40 # M ohms

# I & F implementation dV/dt = - V/RC + I/C
# Using h = 1 ms step size, Euler method

tstop = 5000    # ms
abs_ref = 5 # absolute refractory period 
V_th = 10 # spike threshold (in mV)

# input current
noiseampmax = 5
noiseamp = np.arange(0,noiseampmax+1) # amplitude of added noise

bins = np.linspace(0, 60)   # bin intervals for histogram of interspike interval distribution

for i in range(noiseamp.size):
    V = 0   # mV
    ref = 0 # absolute refractory period counter
    V_trace = []  # voltage trace for plotting
    spiketimes = [] # list of spike times (in ms)
    
    I = I0 + noiseamp[i]*np.random.normal(0, 1, (tstop,)) # nA; Gaussian noise
    
    for t in range(tstop):
      
       if not ref:
           V = V - (V/(R*C)) + (I[t]/C)
       else:
           ref -= 1
           V = 0.2 * V_th # reset voltage
       
       if V > V_th:
           V = 50 # emit spike
           spiketimes += [t]
           ref = abs_ref # set refractory counter
    
       V_trace += [V]
    
    spikeInt = np.diff(spiketimes)  # list of interspike intervals
    
    # plot voltage across time
    plt.plot(V_trace)
    plt.show()
    
    print("Number of spikes = %d\n" %(len(spiketimes)))
    
    # plot histogram of interspike intervals
    lbl = "Noise amplitude = %f" %(noiseamp[i])
    plt.hist(spikeInt, bins, label=lbl)
    plt.title("Histogram of interspike intervals with 50 bins")
    plt.legend()
    plt.show()