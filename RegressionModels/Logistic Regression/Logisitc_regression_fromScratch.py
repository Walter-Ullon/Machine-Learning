
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 19:44:21 2017
@author: walterullon

Logistic Regression from scratch! This version 
"""
#=========================================
#          IMPORT PACKAGES:
#=========================================
import pandas as pd
import numpy as np

#=========================================
#              LOAD DATA:
#=========================================
data = pd.read_csv('E-commerce_data.csv')


#=========================================
#      SET PREDICTORS AND TARGET:
#=========================================
# Get only binary values (0, 1):
X = data.loc[:, data.columns != 'user_action']
X = data[(data['user_action'] == 0) | (data['user_action'] == 1)]

y = X['user_action']

X.drop('user_action', axis=1, inplace=True)

#=========================================
#     NORMALIZE NON-CATEGORICAL DATA:
#=========================================
X['n_products_viewed'] = (X['n_products_viewed'] - X['n_products_viewed'].mean()) / X['n_products_viewed'].std()
X['visit_duration'] = (X['visit_duration'] - X['visit_duration'].mean()) / X['visit_duration'].std()

#=========================================
#       DUMIFFY CATEGORICAL DATA:
#=========================================
X_dummies = pd.get_dummies(X['time_of_day'])

#=========================================
#       DEFINE FINAL PREDICTOR:
#=========================================
X = X.loc[:, X.columns != 'time_of_day']

X_final = pd.concat([X, X_dummies], axis=1)

#=========================================
#       CONVERT TO NP ARRAYS:
#=========================================
X = np.array(X_final)
y = np.array(y)

#=========================================
#       DEFINE SIGMOID FUNCTION:
#=========================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#=========================================
#       DEFINE RANDOM WEIGHTS:
#=========================================
W = np.random.randn(len(X[0]))

#=========================================
#   FORWARD PASS (i.e. Lin. Reg Step):
#=========================================
# bias
b = 0

# Define function:
def forward(x, weights, bias):
    y_hat = x.dot(weights) + bias
    return y_hat


# Perform forward pass:
y_hat = forward(X, W, b)    


#=========================================
#  SCORE PREDICTIONS (pre optimization):
#=========================================
# Define functions:
def classification_rate(Y, YHAT):
    return np.mean(Y == YHAT)

#print("score: ", classification_rate(y, np.round(y_hat)))


#=================================================
#  PERFORM GRADIENT DESCENT TO OPTIMIZE WEIGHTS:
#=================================================
# Define Learning Rate and cost array:
learning_rate = 0.001

# Start Gradient descent:
for i in range(10000):
    y_hat = sigmoid(forward(X, W, b))
    W -= learning_rate*X.T.dot(y_hat - y)
    b -= learning_rate*(y_hat - y).sum()    
    
    
print('final W: ', W)
print("score: ", classification_rate(y, np.round(y_hat)))


















