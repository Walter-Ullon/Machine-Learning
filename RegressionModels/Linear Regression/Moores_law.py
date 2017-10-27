#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:56:41 2017

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
data = pd.read_csv('moore.csv')

#=========================================
#      ASSIGN PREDICTOR & RESPONSE:
#=========================================
x = np.array(data['Date of introduction'])
Y = np.array(data['Transistor count'])
Y = np.log(Y)
#=========================================
#       DEFINE COEFFICIENTS:
#=========================================
n = len(x)
x_bar = np.mean(x)
y_bar = np.mean(Y)
xy_bar = x.dot(Y)/n
x_sq_bar = x.dot(x)/n

# Slope:
a = (xy_bar - np.dot(x_bar, y_bar)) / (x_sq_bar - x_bar**2)
# Intercept:
b = ((y_bar*x_sq_bar) - x_bar*(xy_bar)) / (x_sq_bar - x_bar**2)

#=========================================
#              PREDICTION:
#=========================================
y_hat = a*x + b

#=========================================
#              PLOT:
#=========================================
plt.scatter(x,Y)
plt.plot(x,y_hat, color='red')
plt.show()


#=========================================
#          ANALYZE THE FIT:
#=========================================
# Residual sume of squares:
rss = sum((Y - y_hat)**2)

# Residual sum of square totals:
rss_totals = sum((y_hat - y_bar)**2)

# R-Squared:
R_sq = 1 - (rss / rss_totals)

print('R-squared statistic is: ', R_sq)

'''
How long does it take to double?
    log(transistorcount) = a*year + b
    transistorcount = exp(b) * exp(a*year)
    2*transistorcount = 2 * exp(b) * exp(a*year) = exp(ln(2)) * exp(b) * exp(a * year) = exp(b) * exp(a * year + ln(2))
    a*year2 = a*year1 + ln2
    year2 = year1 + ln2/a
'''
print("time to double:", np.log(2)/a, "years")

