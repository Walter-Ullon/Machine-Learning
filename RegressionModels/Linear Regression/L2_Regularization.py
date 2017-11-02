#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:02:50 2017

@author: walterullon

L2 Regularization Demo
"""
#=========================================
#          IMPORT PACKAGES:
#=========================================
import numpy as np
import matplotlib.pyplot as plt


#=========================================
#         GENERATE SAMPLE DATA:
#=========================================

N = 50
X = np.linspace(0, 10, N)
y = 0.5*X + np.random.randn(N)

# Generate Outliers:
y[-1] += 30
y[-2] += 30

# Plot:
plt.scatter(X, y)
plt.title('Original Data')
plt.show()

#=========================================
#     DEFINE WEIGHTS VECTOR (w/o L2):
#=========================================
# Add a column of ones & Transpose:
X = np.vstack([np.ones(N), X]).T
X_trans = X.T

# Vector of weights:
W = np.linalg.inv(X_trans.dot(X)).dot(X_trans.dot(y))

#=========================================
#        APPROXIMATION (w/o L2):
#=========================================
y_hat = np.dot(X, W)

# Plot:
plt.scatter(X[:,1], y)
plt.plot(X[:,1], y_hat)
plt.title('Best Fit w/o L2 Regularization')
plt.show()



#=========================================
#    DEFINE WEIGHTS VECTOR (using L2):
#=========================================

# penalty:
lamda = 1000

# Vector of weights:
W = np.linalg.inv(lamda*np.eye(2) + X_trans.dot(X)).dot(X_trans.dot(y))


#=========================================
#        APPROXIMATION (using L2):
#=========================================
y_hat = np.dot(X, W)

# Plot:
plt.scatter(X[:,1], y)
plt.plot(X[:,1], y_hat)
plt.title('Best Fit with L2 Regularization')
plt.show()






















