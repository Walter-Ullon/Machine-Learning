#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:52:49 2017

@author: walterullon

POLYNOMIAL REGRESSION

"""

#=========================================
#          IMPORT PACKAGES:
#=========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#=========================================
#              LOAD DATA:
#=========================================
data = pd.read_csv('data_poly.csv', header=None)
data.columns = ['x', 'y']

#=========================================
#      ASSIGN PREDICTOR & RESPONSE:
#=========================================
X = np.array(data['x'])

# We define a new array with a quadratic 'x' term:
X = np.transpose(np.array([X, X**2]))
y = np.array(data['y'])


#=========================================
#                PLOT:
#=========================================
# Original Data:
plt.scatter(X[:,0], y)
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

#=========================================
#                PLOT FIT:
#=========================================
# Original Data:
plt.scatter(X[:,0], y)
plt.plot(sorted(X[:,0]), sorted(y_hat), color = 'red')
plt.show()