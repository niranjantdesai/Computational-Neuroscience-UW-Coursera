# -*- coding: utf-8 -*-
"""
Created on Sun May 07 14:07:07 2017

@author: Niranjan Thakurdesai

Quiz 3 code
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math

import pickle

with open('../data/tuning_2.7.pickle', 'rb') as f:
    data = pickle.load(f)
    
numTrials = 100
numStim = 24
numNeurons = 4
    
stim = data['stim']
neuron = np.empty([numNeurons, numTrials, numStim])
for i in range(numNeurons):
    buf = "neuron%d" % (i+1)
    neuron[i] = data[buf]
    
# Compute mean firing rate for each neuron
meanFiringRate = np.empty([numNeurons, numStim])
for i in range(numNeurons):
    meanFiringRate[i] = np.mean(neuron[i], axis=0)
    
# Plot tuning curve for each neuron
plt.figure(1)
for i in range(numNeurons):
    lbl = "Neuron %d" % (i+1)
    plt.plot(stim, meanFiringRate[i], label=lbl)
plt.xlabel('Stimulus')
plt.ylabel('Mean firing rate')
plt.title('Tuning curve')
plt.legend()
plt.show()


"""
To find out which neuron(s) is(are) not a Poisson neuron, we inspect the Fano
factor for each neuron-stimulus combination. The neuron(s) which are not Poisson
should have a Fano factor that varies with the stimulus.
"""
# Compute Fano factor for each neuron-stimulus combination
var = np.empty([numNeurons, numStim])
fano = np.empty([numNeurons, numStim])
for i in range(numNeurons):
    var[i] = np.var(neuron[i], axis=0)
fano = np.divide(var, meanFiringRate)


with open('../data/pop_coding_2.7.pickle', 'rb') as f:
    pop_coding = pickle.load(f)
numTrials1 = 10
    
"""
Find the population vector by averaging over the 10 trials of the unknown
stimulus
"""
# Compute the maximum average firing rate for each neuron
mfar = np.amax(meanFiringRate, axis=1)

# Compute the population vector for each trial
popVec = np.empty([numTrials1, 2])
r1 = pop_coding['r1']
r2 = pop_coding['r2']
r3 = pop_coding['r3']
r4 = pop_coding['r4']
c1 = pop_coding['c1']
c2 = pop_coding['c2']
c3 = pop_coding['c3']
c4 = pop_coding['c4']
for i in range(numTrials1):
    popVec[i] = (r1[i]/mfar[0])*c1 + (r2[i]/mfar[1])*c2 + (r3[i]/mfar[2])*c3 + \
          (r4[i]/mfar[3])*c4
popVecAvg = np.mean(popVec, axis=0)
theta = math.degrees(math.atan2(popVecAvg[1],popVecAvg[0]))