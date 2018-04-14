# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 15:43:34 2018

@author: yj7082126
"""

import os
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = pd.read_csv("data/ex2data2.txt", header=None).as_matrix()
X = data[:, 0:2]
y = data[:, 2:]

#%% Function Definition

def sigmoid(X, theta):
    h = X.dot(theta)
    return 1 / (1 + np.exp(-h))

def mapFeature(X, deg):
    out = np.ones((X.shape[0], (int)((deg+1)*(deg+2)/2)))
    x = 0
    for i in range(deg+1):
        for j in range(i+1):
            out[:,x] = np.multiply(np.power(X[:,0], i-j), np.power(X[:,1], j))
            x += 1
    return out
    
def costF(theta, X, y, lambd):
    m = len(y)
    sig = sigmoid(X, theta)
    J1 = np.transpose(y) @ np.log(sig)
    J2 = np.transpose(1-y) @ np.log(1-sig)
    J = (-1/m) * np.sum(J1+J2)   
    J = J + (lambd/(2*m))*sum(np.square(theta[1:]))  
    
    grad = ((1/m) * (sig - y.flatten()).dot(X))
    grad2 = grad + ((lambd/m)*theta)
    grad2[0] = grad[0]
    return J, grad2

#%% Plot Data
    
print("""Plotting data with o indicating (y == 1) examples 
      and x indicating (y == 0) examples.\n""")

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[(y==1).flatten()][:,0],X[(y==1).flatten()][:,1], marker="o", label="Admitted")
ax.scatter(X[(y==0).flatten()][:,0],X[(y==0).flatten()][:,1], marker="x", label="Not Admitted")
plt.title("Scatterplot between Microchip Test Scores")
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend()
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Gradient Descent Settings

#Regularization parameter lambda to 1, degree to 6
degree = 6
lambd = 1
#Add Polynomial Feature.
X = mapFeature(X, degree) 
[m, n] = np.shape(X)
theta = np.zeros(n)

#%% Cost Function Regularization & Gradient Descent

cost, grad = costF(theta, X, y, lambd)
print("Cost at initial theta (zeros) : ", cost, "\n")
print("Expected cost value (approx) : 0.693 \n")
print("Gradient at initial theta (zeros) - first five: \n", grad[:5], "\n")
print("Expected gradients (approx) :\n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n")

input("Program Paused. Press enter to continue. \n")

#%% Cost Function Regularization & Gradient Descent #2

theta = np.ones(n)
lambd = 10
cost, grad = costF(theta, X, y, lambd)
print("Cost at initial theta (zeros) : ", cost, "\n")
print("Expected cost value (approx) : 3.16 \n")
print("Gradient at initial theta (zeros) - first five: \n", grad[:5], "\n")
print("Expected gradients (approx) :\n 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n")

input("Program Paused. Press enter to continue. \n")

#%% Optimization

theta = np.zeros(n)
lambd = 1

thet_res = minimize(costF, theta, method='Newton-CG', args=(X, y, lambd), 
                    jac=True, options={'maxiter':400, 'disp':True})

if thet_res['success'] == True:
    print("Cost at theta found by optimization : ", thet_res['fun'], '\n')
    print("Expected cost (approx) : 0.529 \n")
    print("Theta found by optimization - first five : \n", thet_res['x'][:5], "\n")
    print("Expected theta (approx) :\n 1.273\n 0.625\n 1.181\n -2.020\n -0.917\n")
else:
    print("Optimization failed.\n")    

input("Program Paused. Press enter to continue. \n")

#%% Plot Boundary

u = np.arange(-1, 1.5, 0.05)
v = np.arange(-1, 1.5, 0.05)

m = len(u)
z = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        z[i, j] = mapFeature(np.array([[u[i], v[j]]]), 6).dot(thet_res['x'])
        
u2, v2 = np.meshgrid(u, v) 

fig, ax = plt.subplots(figsize=(12,8))
posX = X[(y==1).flatten()][ :, 1:3]
negX = X[(y==0).flatten()][ :, 1:3]
ax.scatter(posX[:,0], posX[:,1], marker="o", label="Admitted")
ax.scatter(negX[:,0], negX[:,1], marker="x", label="Not Admitted")
cs = ax.contour(u2, v2, z, levels=[0])
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend()
plt.show()

