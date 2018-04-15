# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 10:00:54 2018

@author: yj7082126
"""

import os
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = loadmat('data/ex5data1.mat')
X = data['X']
Xtest = data['Xtest']
Xval = data['Xval']
y = data['y']
ytest = data['ytest']
yval = data['yval']

#%% Function Definition

def linReg_costF(theta, X, y, lambd):
    m = len(y)
    h = X.dot(theta)
    sqrterror = np.square(h - y.flatten())
    J = (1/(2*m)) * sum(sqrterror)
    J = J + (lambd/(2*m))*sum(np.square(theta[1:]))
    
    grad = ((1/m) * (h - y.flatten()).dot(X))
    grad2 = grad + ((lambd/m)*theta)
    grad2[0] = grad[0]
    return J, grad2

def train_linReg(X, y, lambd):
    init_theta = np.zeros(X.shape[1])
    tt = minimize(linReg_costF, init_theta, 
                  args=(X, y, lambd),  
                  jac=True, 
                  options={'maxiter': 200, 'disp': True})
    return tt

def learnCurve(X, y, Xval, yval, lambd):
    error_train = np.zeros(len(y))
    error_val = np.zeros(len(y))
    
    for i in range(len(y)):
        theta = train_linReg(X[:i+1], y[:i+1], lambd)['x']
        error_train[i] = linReg_costF(theta, X[:i+1], y[:i+1], 0)[0]
        error_val[i] = linReg_costF(theta, Xval, yval, 0)[0]
        
    return error_train, error_val

def polyFeature(X, p):
    X_poly = np.zeros((len(X), p))
    for i in range(p):
        X_poly[:,i] = np.power(X, i+1).flatten()
    return X_poly
        
def fNorm(X):
    mu = np.mean(X, axis=0)
    sig = np.std(X, ddof=1, axis=0)
    X_norm = (X - mu) / sig
    return X_norm, mu, sig

def plotFit(theta, mu, sig):
    n = 50
    xvals = np.linspace(-55, 43, n).reshape((-1, 1))
    
    #xmat = np.ones((n, 1))
    #xmat = np.insert(xmat, xmat.shape[1], np.transpose(xvals), axis=1)
    xmat = polyFeature(xvals, len(theta)-1)
    xmat = np.hstack((np.ones((len(xmat), 1)), xmat))
    
    xmat[:,1:] = (xmat[:,1:] - mu) / sig
    
    plt.figure(figsize=(6,4))
    plt.scatter(X, y, marker='x')
    plt.plot(xvals, xmat @ theta)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()
 
def validationCurve(X, y, Xval, yval, lambd_vec):
    error_train = np.zeros(len(lambd_vec))
    error_val = np.zeros(len(lambd_vec))
    
    for i in range(len(lambd_vec)):
        lambd = lambd_vec[i]
        theta = train_linReg(X, y, lambd)['x']
        error_train[i] = linReg_costF(theta, X, y, 0)[0]
        error_val[i] = linReg_costF(theta, Xval, yval, 0)[0]
    
    return error_train, error_val
    
#%% Plot Data
 
print('Visualizing Data... \n')
plt.figure(figsize=(6,4))
plt.scatter(X, y, marker='x')
plt.title("Scatterplot between Water Level & Dam Flow ")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Settings

[m, n] = X.shape
X2 = np.append(np.ones((m, 1)), X, axis=1)
theta = np.ones(n+1)
lambd = 1

#%% Regularized Linear Regression

print('Regularized Linear Regression Cost & Gradient... \n')

J, grad = linReg_costF(theta, X2, y, lambd)
print('Cost at theta = [1 ; 1]: ', J, ' \n')
print('This value should be about 303.993192. \n')

print('Gradient at theta = [1 ; 1]: ', grad, '\n')
print('This value should be about [-15.303016; 598.250744 ] \n')

input("Program Paused. Press enter to continue. \n")

#%% Train Linear Regression

print('Train Linear Regression... \n')
lambd = 0
theta = train_linReg(X2, y, lambd)

plt.figure(figsize=(6,4))
plt.scatter(X, y, marker='x')
plt.plot(X, X2 @ theta['x'])
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Learning Curve

print('Learning Curve for Linear Regression... \n')

Xval2 = np.append(np.ones((len(Xval), 1)), Xval, axis=1)
[error_train, error_csv] = learnCurve(X2, y, Xval2, yval, 0)

print(pd.DataFrame({
        'Train Error': error_train, 
        'Cross Validation Error': error_csv
        })[['Train Error', 'Cross Validation Error']])
    
plt.figure(figsize=(6,4))
plt.plot(np.array(range(len(y))), error_train)
plt.plot(np.array(range(len(y))), error_csv)
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Feature Mapping for Polynomial Regression

print('Feature Mapping for Polynomial Regression... \n')
p = 8

Xp = polyFeature(X, p)
Xp, mu, sig = fNorm(Xp)
Xp = np.append(np.ones((len(Xp), 1)), Xp, axis=1)

Xptest = polyFeature(Xtest, p)
Xptest = (Xptest - mu) / sig
Xptest = np.append(np.ones((len(Xptest), 1)), Xptest, axis=1)

Xpval = polyFeature(Xval, p)
Xpval = (Xpval - mu) / sig
Xpval = np.append(np.ones((len(Xpval), 1)), Xpval, axis=1)

print('Normalized Training Example 1: \n')
print(Xp[1,:])

input("Program Paused. Press enter to continue. \n")

#%%  Train Polynomial Regression

print('Train Polynomial Regression...\n')

lambd = 1
theta = train_linReg(Xp, y, lambd)

plotFit(theta['x'], mu, sig)

input("Program Paused. Press enter to continue. \n")

#%% Learning Curve for Polynomial Regression

print('Learning Curve for Polynomial Regression... \n')

[error_train, error_csv] = learnCurve(Xp, y, Xpval, yval, lambd)

plt.figure(figsize=(6,4))
plt.plot(np.array(range(len(y))), error_train)
plt.plot(np.array(range(len(y))), error_csv)
plt.show()

print('Polynomial Regression (lambda = %d) \n' % (lambd))
print(pd.DataFrame({
        'Test Error': error_train, 
        'Cross Validation Error': error_csv
        })[['Test Error', 'Cross Validation Error']])
  
input("Program Paused. Press enter to continue. \n")

#%% Validation for selecting lambda

lambd_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3])
[error_train, error_csv] = validationCurve(Xp, y, Xpval, yval, lambd_vec)

plt.figure(figsize=(6,4))
plt.plot(lambd_vec, error_train)
plt.plot(lambd_vec, error_csv)
plt.show()

print(pd.DataFrame({
        'lambda' : lambd_vec, 
        'Train Error': error_train, 
        'Validation Error': error_csv
        })[['lambda', 'Train Error', 'Validation Error']])
    