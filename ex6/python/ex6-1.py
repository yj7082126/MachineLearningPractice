# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:29:43 2018

@author: user
"""

import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = loadmat('data/ex6data1.mat')
X = data['X']
y = data['y']

#%% Function Definition

def visualizeBoundaryLinear(X, y, model):
    weights = model.coef_[0]
    intercept = model.intercept_[0]
    
    pos = (y == 1).flatten()
    neg = (y == 0).flatten()
    
    Xs = np.linspace(X.min(), X.max(), 100)
    ys = - (weights[0] * Xs + intercept) / weights[1]
    
    fig, ax = plt.subplots(figsize=(12,8))
    plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black', marker='o')
    plt.plot(Xs, ys)
    plt.show()

def visualizeBoundary(X, y, model):   
    X1 = np.linspace(X.min(axis=0)[0], X.max(axis=0)[0], 100)
    X2 = np.linspace(X.min(axis=0)[1], X.max(axis=0)[1], 100)
    X1, X2 = np.meshgrid(X1, X2)
    
    Xs = np.append(X1.reshape(-1, 1), X2.reshape(-1, 1), axis=1)
    ys = svm.predict(Xs).reshape(X1.shape)
    
    pos = (y == 1).flatten()
    neg = (y == 0).flatten()
    
    fig, ax = plt.subplots(figsize=(12,8))
    plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black', marker='o')
    plt.contour(X1, X2, ys)
    plt.show()
    
def dataset_params(X, y, Xval, yval):
    Cv = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    sigv = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    scores = np.zeros((8, 8))
    
    for i in range(8):
        for j in range(8):
            svm = SVC(kernel='rbf', C=Cv[i], gamma=sigv[j])
            svm.fit(X, y.flatten())
            scores[i,j] = accuracy_score(yval, svm.predict(Xval))
    
    max_c_ind, max_s_ind = np.unravel_index(scores.argmax(), scores.shape)
    return (Cv[max_c_ind], sigv[max_s_ind])
    
def gauss(x1, x2, sigma):
    return np.exp(- (np.square(np.linalg.norm(x1 - x2)).sum())/(2 * (np.square(sigma))))

#%% Plot Data
    
print('Visualizing Data \n')
pos = (y == 1).flatten()
neg = (y == 0).flatten()

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black', marker='o')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Training & Visualizing Linear SVM

print('Training Linear SVM... \n')
C = 1
svm = SVC(kernel='linear', C=C)
svm.fit(X, y.flatten())

visualizeBoundaryLinear(X, y, svm)

input("Program Paused. Press enter to continue. \n")

#%% Implementing Gaussian Kernel

print('Evaluating the Gaussian Kernel \n')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gauss(x1, x2, sigma)

print('Gaussian Kernel between x1= [1; 2; 1], x2 = [0; 4; -1], sigma = 2: ', 
      sim, '\n')
print('This value should be about 0.32465 \n')

input("Program Paused. Press enter to continue. \n")

#%% Initialization #2

data = loadmat('data/ex6data2.mat')
X = data['X']
y = data['y']

#%% Plot Data #2
    
print('Visualizing Data #2\n')
pos = (y == 1).flatten()
neg = (y == 0).flatten()

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black', marker='o')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Training & Visualizing RBF Kernel SVM

print('Training RBF Kernel SVM \n')
C = 30
sigma = 30
svm = SVC(kernel='rbf', C=C, gamma=sigma)
svm.fit(X, y.flatten())

visualizeBoundary(X, y, svm)

input("Program Paused. Press enter to continue. \n")

#%% Initialization #3

print('Loading Data \n')
data = loadmat('data/ex6data3.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']

#%% Plot Data #3

print('Visualizing Data \n')
pos = (y == 1).flatten()
neg = (y == 0).flatten()

fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(X[pos, 0], X[pos, 1], c='black', marker='+')
plt.scatter(X[neg, 0], X[neg, 1], c='yellow', edgecolors='black', marker='o')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Training & Visualizing RBF Kernel SVM

C, sigma = dataset_params(X, y, Xval, yval)
svm = SVC(kernel='rbf', C=C, gamma=sigma)
svm.fit(X, y.flatten())

visualizeBoundary(X, y, svm)

input("Program Paused. Press enter to continue. \n")