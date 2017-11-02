#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 09:37:25 2017

@author: walterullon
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
'''
Data from: http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/
           frames/mlr02.html
'''
data = pd.read_csv('mlr02.csv')
data.columns = ['Blood Pressure', 'Age', 'Weight']

#=========================================
#      CHECK LINEAR RELATIONSHIPS:
#=========================================
# Age vs BP
plt.scatter(data['Age'], data['Blood Pressure'])
plt.title('Age vs. Blood Pressure')
plt.show()

# Weight vs BP
plt.scatter(data['Weight'], data['Blood Pressure'])
plt.title('Weight vs. Blood Pressure')
plt.show()

#=========================================
#       SET PREDICTOR AND TARGET:
#=========================================
X = np.array(data[['Age', 'Weight']])
y = np.array(data['Blood Pressure'])

#=========================================
#       CALCULATE WEIGHTS VECTOR:
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






























