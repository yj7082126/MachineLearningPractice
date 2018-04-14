# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 12:39:31 2018

@author: yj7082126

@desc: This program implements Linear Regression with single variable.
"""

import os
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = pd.read_csv("data/ex1data1.txt", header=None).as_matrix()
X = data[:, 0:1]
y = data[:, 1:2]

#%% Function definition

def costF(theta, X, y):
    m = len(y)
    h = X.dot(theta)
    sqrterror = np.square(h - y.flatten())
    J = (1/(2*m)) * sum(sqrterror)
    return J
    
def gradF(theta, X, y, alpha, iterations):
    m = len(y)
    J_hist = np.zeros((iterations, 1))
    for it in range(iterations):
        h = X.dot(theta)
        j0 = alpha * (1/m) * sum(h - y.flatten())
        j1 = alpha * (1/m) * X[:,1].dot(h-y.flatten())
        t0 = theta[0] - j0
        t1 = theta[1] - j1
        theta = np.array([t0,t1])
        J_hist[it] = costF(theta, X, y)
    return theta, J_hist

def predict(theta, val):
    val2 = np.array([1, val])
    return val2.dot(theta)

#%% Plot Data
    
print("Plotting Data ... \n")
plt.scatter(X, y, marker='x')
plt.title("Scatterplot between Profit & Population")
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Gradient Descent Settings

[m, n] = np.shape(X)

X = np.append(np.ones((m, 1)), X, axis=1).astype(float)
theta = np.zeros(n+1)

iterations = 1500;
alpha = 0.01

#%% Cost Function

print("Testing Cost Function ... \n")

J = costF(theta, X, y)
print("With theta = [0 ; 0] \nCost computed : ", J, "\n")
print("Expected cost value (approx) 32.07 \n")

J = costF(np.array([-1, 2]), X, y)
print("With theta = [-1 ; 2] \nCost computed : ", J, "\n")
print("Expected cost value (approx) 54.24 \n")

input("Program Paused. Press enter to continue. \n")

#%% Gradient Descent

print("Running Gradient Descent ... \n")

theta, J_hist = gradF(theta, X, y, alpha, iterations)
print("Theta found by gradient descent : \n", theta, "\n")
print("Expected theta values (approx) : \n -3.6303\n 1.1664\n")

input("Program Paused. Press enter to continue. \n")

#%% Plot Linear Fit

scatter = plt.scatter(X[:,1], y, marker='x', label='Training Data')
plt.title("Scatterplot between Profit & Population")
plt.xlabel('Profit in $10,000s')
plt.ylabel('Population of City in 10,000s')
line = plt.plot(X[:,1], (X @ theta).flatten(), color="r", label='Linear Regression')
plt.legend()
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Predict Value for diff population sizes

print("""For population = 35,000, we predict profit of %d""" 
      % (predict(theta, 35000)))

print("""For population = 70,000, we predict profit of %d""" 
      % (predict(theta, 70000)))

input("Program Paused. Press enter to continue. \n")

#%% Visualize J

print("Visualizing J(Theta0, Theta1)")

# Grid to calculate J
t0val = np.arange(-10, 10, 0.2)
t1val = np.arange(-1, 4, 0.05)

m = len(t0val)
Jval = np.zeros((m, m))

t0val2, t1val2 = np.meshgrid(t0val, t1val) 

for i in range(m):
    for j in range(m):
        t = np.array([t0val2[i, j], t1val2[i, j]])
        Jval[i, j] = costF(t, X, y)

# Surface Plot              
fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(t0val2, t1val2, Jval)
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Cost')
plt.show()

# Contour Plot
fig2, ax2 = plt.subplots()
cs = ax2.contour(t0val2, t1val2, Jval, levels=np.logspace(-2, 3, num=20))
ax2.text(theta[0], theta[1], 'x')
#plt.clabel(cs, inline=1)
#%%
