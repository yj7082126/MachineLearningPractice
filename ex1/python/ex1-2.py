# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 14:25:40 2018

@author: yj7082126

@desc: This program implements Linear Regression with multiple variables.
"""

import os
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = pd.read_csv("data/ex1data2.txt", header=None).as_matrix()
X = data[:, 0:2]
y = data[:, 2:]

print("Loading data ... \n")
print("First 10 examples from the dataset : \n")
for i in range(10):
    print("x = ", X[i], ", y = ", y[i])

input("Program Paused. Press enter to continue. \n")

#%% Function definition

def fNorm(X):
    mu = np.mean(X, axis=0)
    sig = np.std(X, axis=0)
    X_norm = (X - mu) / sig
    return X_norm, mu, sig
    
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
        theta = theta - alpha*(1/m)*np.transpose(X).dot(h - y.flatten())
        J_hist[it] = costF(theta, X, y)
    return theta, J_hist

def predict(theta, mu, sig, val):
    val2 = np.insert((val - mu) / sig, 0, 1)
    return val2.dot(theta)

#%% Gradient Descent Settings
 
theta = np.zeros(np.size(data, axis=1))

iterations = 500
alpha = 0.01

#%%  Normalize Feature & Gradient Descent

print("Normalizing Feature ...\n")
X2, mu, sig = fNorm(X)
X = np.append(np.ones((len(data), 1)), X2, axis=1).astype(float)

print("Running Gradient Descent ...\n")
theta, J_hist = gradF(theta, X, y, alpha, iterations)

print("Theta found by gradient descent : \n", theta, "\n")

input("Program Paused. Press enter to continue. \n")

#%% Plotting Convergence Graph

fig, ax = plt.subplots(figsize=(12,8))
plt.title("Convergence graph of J")
ax.plot(J_hist)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost J')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Predicting Value
pred = np.array([1650, 3])
price = predict(theta, mu, sig, pred)

print(""" Predicted price of a 1650 sq-ft, 3 br house
          (using gradient descent) : \n""", price, "\n")
