# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:35:46 2018

@author: user
"""

import os
import os.path as path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from math import sqrt
from scipy.io import loadmat

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = loadmat('data/ex3data1.mat')
X = data['X']
y = data['y']
#y[y > 9] = 0

mat = loadmat('data/ex3weights.mat')
Theta1 = mat['Theta1']
Theta2 = mat['Theta2']
#%% Function Definition

def sigmoid(X, theta):
    h = X.dot(theta)
    return 1 / (1 + np.exp(-h))

def predict(Theta1, Theta2, X):
    m = len(X)
    X = np.append(np.ones((m, 1)), X, axis=1).astype(float)
    Z2 = sigmoid(X, np.transpose(Theta1))
    Z2 = np.append(np.ones((m, 1)), Z2, axis=1).astype(float)
    h = sigmoid(Z2, np.transpose(Theta2))
    return [x+1 for x in np.argmax(h, axis=1)]

#%% Plot Data

print("Visualizing Data ...\n")
m = int(sqrt(len(X[0])))
ra = np.random.randint(0, 5000, size=1)[0]
plt.imshow(np.transpose(np.reshape(X[ra,:], (m, m))))
plt.show()
print(y[ra])

input("Program Paused. Press enter to continue. \n")

#%% Settings

inputSize = 400
hiddenSize = 25
num = 10

#%% Prediction w/ Preloaded NN

pred = predict(Theta1, Theta2, X)

cnt = 0
for i in range(m):
    if pred[i] == y[i][0]:
        cnt += 1

print("Training Set Accuracy: %f %%" % (cnt*100/m))    