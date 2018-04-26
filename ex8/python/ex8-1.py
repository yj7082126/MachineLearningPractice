# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:17:36 2018

@author: user
"""

import os
import os.path as path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = scipy.io.loadmat('data/ex8data1.mat')
X = np.array(data['X'])
Xval = np.array(data['Xval'])
yval = np.array(data['yval']).flatten()

#%% Function definition

def visualizeFit(X, *outlier):
    mu = np.mean(X, axis=0)
    sig2 = np.var(X, axis=0)
    
    maxval = np.max((np.max(X, axis=0)/5).round() * 5)
    meshval = np.arange(0, maxval, 0.5)
    X1, X2 = np.meshgrid(meshval, meshval)
    Z = np.hstack((X1.reshape((-1,1)), X2.reshape((-1, 1))))
    Z = multivariate_normal.pdf(Z, mu, sig2).reshape(np.shape(X1))
    
    level = np.array([10**-15, 10**-13, 10**-11, 10**-9, 10**-7, 
                      10**-5, 10**-3, 10**-1])
    fig, ax = plt.subplots(figsize=(12,12))
    ax.scatter(X[:,0], X[:,1], marker='x')
    ax.contour(X1, X2, Z, level)
    
    if outlier != None:
        ax.scatter(X[outlier, 0], X[outlier, 1], marker = 'o', color='r')
    plt.show()
  
def selectThreshold(X, Xval, yval):
    mu = np.mean(X, axis=0)
    sig2 = np.var(X, axis=0)
    
    pval = multivariate_normal.pdf(Xval, mu, sig2)
   
    optimF1 = 0
    optimepsil = 0
    for i in range(1000):
        epsil = (1 - (i/1000))*np.min(pval) + (i/1000)*np.max(pval)
        predict = (pval < epsil)
        
        tp = len(np.intersect1d(np.where(yval == True), 
                            np.where(predict == True)))
        
        fp = len(np.intersect1d(np.where(yval == True), 
                            np.where(predict == False)))
        fn = len(np.intersect1d(np.where(yval == False), 
                            np.where(predict == True)))
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        if (precision + recall == 0):
            F1 = 0
        else:
            F1 = (2 * precision * recall) / (precision + recall)
        
        if F1 > optimF1:
            optimF1 = F1
            optimepsil = epsil
            
    return (optimF1, optimepsil)
    
#%% Plot Data

print("Plotting Data ... \n")
plt.scatter(X[:,0], X[:,1], marker='x')
plt.title("Scatterplot between Latency & Throughput")
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Estimate dataset (Gaussian)

mu = np.mean(X, axis=0)
sig2 = np.var(X, axis=0)

visualizeFit(X)

input("Program Paused. Press enter to continue. \n")

#%% Find Outliers

F1, epsil = selectThreshold(X, Xval, yval)

print("Best epsilon found using cross-validation : ", epsil, " \n")
print("Best F1 on Cross Validation Set : ", F1, " \n")
print('   (you should see a value epsilon of about 8.99e-05)\n');
print('   (you should see a Best F1 value of  0.875000)\n\n');

p = multivariate_normal.pdf(X, mu, sig2)
outlier = np.where(p < epsil)

visualizeFit(X, outlier)

#%% Initialization #2

data = scipy.io.loadmat('data/ex8data2.mat')
X = np.array(data['X'])
Xval = np.array(data['Xval'])
yval = np.array(data['yval']).flatten()

#%% Estimate dataset #2 (Gaussian)

mu = np.mean(X, axis=0)
sig2 = np.var(X, axis=0)

p = multivariate_normal.pdf(X, mu, sig2)
pval = multivariate_normal.pdf(Xval, mu, sig2)

F1, epsil = selectThreshold(X, Xval, yval)

print("Best epsilon found using cross-validation : ", epsil, " \n")
print("Best F1 on Cross Validation Set : ", F1, " \n")
print('   (you should see a value epsilon of about 1.38e-18)\n');
print('   (you should see a Best F1 value of  0.615385)\n\n');