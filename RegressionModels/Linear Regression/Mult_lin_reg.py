#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:25:12 2017

@author: walterullon

MULTI DIMENSIONAL LINEAR REGRESSION

"""

#=========================================
#          IMPORT PACKAGES:
#=========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D


#=========================================
#              LOAD DATA:
#=========================================
data = pd.read_csv('data_2d.csv', header=None)
data.columns = ['x1', 'x2', 'y']


#=========================================
#      ASSIGN PREDICTOR & RESPONSE:
#=========================================
X = np.array(data[['x1', 'x2']])
y = np.array(data['y'])


#=========================================
#                PLOT:
#=========================================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], y)
plt.show()

#=========================================
#       DEFINE WEIGHTS VECTOR:
#=========================================
# X transpose:
X_trans = np.transpose(X)

# Vector of weights:
W = np.linalg.inv(X_trans.dot(X)).dot(X_trans.dot(y))


#=========================================
#          APPROXIMATION:
#=========================================
y_hat = np.dot(X, W)


#=========================================
#          ANALYZE THE FIT:
#=========================================
# Residual sume of squares:
rss = sum((y - y_hat)**2)

# Residual sum of square totals:
y_bar = np.mean(y)
rss_totals = sum((y_hat - y_bar)**2)

# R-Squared:
R_sq = 1 - (rss / rss_totals)

print('R-squared statistic is: ', R_sq)


























