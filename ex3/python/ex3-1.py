# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 18:35:32 2018

@author: yj7082126
"""

import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import sqrt
from scipy.optimize import minimize

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = loadmat('data/ex3data1.mat')
X = data['X']
y = data['y']
y[y > 9] = 0

#%% Function Definition
def sigmoid(X, theta):
    h = X.dot(theta)
    return 1 / (1 + np.exp(-h))

def costF(theta, X, y, lambd):  
    m = len(y)
    sig = sigmoid(X, theta)
    J1 = np.transpose(y) @ np.log(sig)
    J2 = np.transpose(1-y) @ np.log(1-sig)
    J = (-1/len(y)) * np.sum(J1+J2)
    J = J + (lambd/(2*m))*sum(np.square(theta[1:]))  

    grad = ((1/m) * (sig - y.flatten()).dot(X))
    grad2 = grad + ((lambd/m)*theta)
    grad2[0] = grad[0]      
    return J, grad2

#%% Plot Data

print("Visualizing Data ...\n")
m = int(sqrt(len(X[0])))
ra = np.random.randint(0, 5000, size=1)[0]
plt.imshow(np.transpose(np.reshape(X[ra,:], (m, m))))
plt.show()
print(y[ra])

input("Program Paused. Press enter to continue. \n")

#%% Vectorize Logistic Regression (Test)

print("Testing costfunction with regularization ...\n")
theta_t = np.array([-2, -1, 1, 2])
X_t = np.array([[1, 0.1, 0.6, 1.1], [1, 0.2, 0.7, 1.2], [1, 0.3, 0.8, 1.3], 
                [1, 0.4, 0.9, 1.4], [1, 0.5, 1, 1.5]])
y_t = np.array([[1], [0], [1], [0], [1]])
lambd = 3

cost, grad = costF(theta_t, X_t, y_t, lambd)
print("Cost at initial theta : ", cost, "\n")
print("Expected cost value (approx) : 2.534819 \n")
print("Gradient at initial theta : \n", grad, "\n")
print("Expected gradients (approx) :\n 0.146561\n -0.548558\n 0.724722\n 1.398003\n")

input("Program Paused. Press enter to continue. \n")

#%% Settings

[m, n] = np.shape(X)

K = 10
lambd = 1

X = np.append(np.ones((len(X), 1)), X, axis=1).astype(float)
all_theta = np.zeros((K, n+1))

#%% Minimize: One-vs-All Training

print("Training One-vs-All Logistic Regression...\n")

for i in range(K):
    theta = np.zeros(n+1)
    tt = minimize(costF, theta, 
                  args=(X, (y == (i)).astype(np.int), lambd), 
                  method='BFGs', jac=True, tol=1e-6,
                  options={'maxiter': 400, 'disp':True})
    if tt['success'] == True:
        all_theta[i,:] = tt['x']
    else:
        print("Optimization failed.\n")  
        break

input("Program Paused. Press enter to continue. \n")
  
#%% Predict for One-vs-All

h = sigmoid(X, np.transpose(all_theta))
p = np.reshape(np.argmax(h, axis=1), 5000, 1)

cnt = 0
for i in range(m):
    if p[i] == y[i][0]:
        cnt += 1

print("Training Set Accuracy: %f %%" % (cnt*100/m))    