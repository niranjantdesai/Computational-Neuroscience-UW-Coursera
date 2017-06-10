# -*- coding: utf-8 -*-
"""
Created on Sat Jun 03 23:38:43 2017

@author: Niranjan Thakurdesai
"""

import numpy as np
from numpy import linalg as LA

Q = np.matrix('0.15 0.1; 0.1 0.12')     # input correlation matrix
print("Q = %s" %(Q))

ev, e = LA.eig(Q)
print("ev = %s" %(ev))
print("e = %s" %(e))

ratio = e[0,np.argmax(ev)]/e[1,np.argmax(ev)]
print('Ratio = %f' %(ratio))