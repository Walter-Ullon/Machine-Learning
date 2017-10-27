
#=========================================
#          IMPORT PACKAGES:
#=========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#=========================================
#              LOAD DATA:
#=========================================
data = pd.read_csv('data-1d.csv', header=None)
data.columns = ['x', 'y']

#=========================================
#      ASSIGN PREDICTOR & RESPONSE:
#=========================================
x = np.array(data['x'])
Y = np.array(data['y'])

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
#            APPROXIMATION::
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

# Residual sum os square totals:
rss_totals = sum((y_hat - y_bar)**2)

# R-Squared:
R_sq = 1 - (rss / rss_totals)

print('R-squared statistic is: ', R_sq)











