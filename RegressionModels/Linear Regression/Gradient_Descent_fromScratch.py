#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 20:13:38 2017

@author: walterullon

'''
Gradient descent code implemented from scratch!
'''
"""

#=========================================
#          IMPORT PACKAGES:
#=========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#=========================================
#              LOAD DATA:
#=========================================
data = pd.read_csv('data_2d.csv', header=None)
data.columns = ['x1', 'x2', 'y']


#=========================================
#      ASSIGN PREDICTOR & RESPONSE:
#=========================================
# Number of observations:
N = len(data['x1'])

X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])

#=========================================
#                PLOT:
#=========================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['x1'], data['x2'], y)
plt.show()

#=========================================
# IMPLEMENT GRADIENT DESCENT ALGORITHM:
#=========================================
# Number of features:
D = len(X[1,:])

# [1] Create 'weights' vector with random values from an uniform distribution:
W = np.random.uniform(0, 1, D)

# [2] Choose learning rate:
eta = 10**(-7)
 
# [3] Travel down W's gradient to find the minimum. Iterate:
for i in range(10**4):
    # [4] Calculate y-hat:
    y_hat = X.dot(W)
    
    # [5] Calculate absolute 'error':
    delta = y_hat - y
    
    # [6] Recalculate W:
    W = W - eta*X.T.dot(delta)


#=========================================
#             ANALYZE FIT:
#=========================================
# Residual sum of squares:
rss = sum((y - y_hat)**2)
# Residual sum of squares totals:
rss_totals = sum((y_hat - np.mean(y))**2)

# R-Squared statistic:
R_sq = 1 - (rss/rss_totals)

print('The R-square statistic is: ' + str(R_sq))




















