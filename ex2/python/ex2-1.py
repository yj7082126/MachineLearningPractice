# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:58:01 2018

@author: yj7082126

@desc: This program implements Logistic Regression
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%% Initialization

data = pd.read_csv("ex2data1.txt", header=None).as_matrix()
X = data[:, 0:2]
y = data[:, 2:]

#%% Function definition

def sigmoid(X, theta):
    h = X.dot(theta)
    return 1 / (1 + np.exp(-h))

def costF(theta, X, y):
    m = len(y)
    sig = sigmoid(X, theta)
    J1 = np.transpose(y) @ np.log(sig)
    J2 = np.transpose(1-y) @ np.log(1-sig)
    J = (-1/m) * np.sum(J1+J2)
    
    grad = ((1/m) * (sig - y.flatten()).dot(X))
    
    return J, grad

#%% Plot Data

print("""Plotting data with o indicating (y == 1) examples 
      and x indicating (y == 0) examples.\n""")
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[(y==1).flatten()][:,0],X[(y==1).flatten()][:,1], marker="o", label="Admitted")
ax.scatter(X[(y==0).flatten()][:,0],X[(y==0).flatten()][:,1], marker="x", label="Not Admitted")
plt.title("Scatterplot between Exam Scores")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Gradient Descent Settings

[m, n] = np.shape(X)

# Add Intercept term to X
X = np.append(np.ones((m, 1)), X, axis=1).astype(float)
theta = np.zeros(n+1)

#%% Cost Function & Gradient Descent

cost, grad = costF(theta, X, y)
print("Cost at initial theta (zeros) : ", cost, "\n")
print("Expected cost value (approx) : 0.693 \n")
print("Gradient at initial theta (zeros) : \n", grad, "\n")
print("Expected gradients (approx) :\n -0.1000\n -12.0092\n -11.2628\n")

test_theta = np.array([-24, 0.2, 0.2])

cost, grad = costF(test_theta, X, y)

print("Cost at initial theta (zeros) : ", cost, "\n")
print("Expected cost value (approx) : 0.218 \n")
print("Gradient at initial theta (zeros) : \n", grad, "\n")
print("Expected gradients (approx) :\n 0.043\n 2.566\n 2.647\n")

input("Program Paused. Press enter to continue. \n")

#%% Optimization

thet_res = minimize(costF, theta, method='Newton-CG', args=(X,y), jac=True,
               options={'maxiter':400, 'disp':True})
if thet_res['success'] == True:
    print("Cost at theta found by optimization : ", thet_res['fun'], '\n')
    print("Expected cost (approx) : 0.203 \n")
    print("Theta found by optimization : \n", thet_res['x'], "\n")
    print("Expected theta (approx) : \n 25.161\n 0.206\n 0.201\n")
else:
    print("Optimization failed.\n")    

input("Program Paused. Press enter to continue. \n")
    
#%% Plot Boundary

theta = thet_res['x']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[(y==1).flatten()][:,1],X[(y==1).flatten()][:,2], marker="o", label="Admitted")
ax.scatter(X[(y==0).flatten()][:,1],X[(y==0).flatten()][:,2], marker="x", label="Not Admitted")
line_x = [np.min(X[:,1])-2, np.max(X[:,2])+2]
line_y = np.multiply((-1/theta[2]),np.add(np.multiply(theta[1], line_x), theta[0]))
ax.plot(line_x, line_y)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

input("Program Paused. Press enter to continue. \n")
