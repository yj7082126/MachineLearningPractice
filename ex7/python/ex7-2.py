# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 21:56:26 2018

@author: user
"""

import os
import os.path as path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import scipy.io
import matplotlib.pyplot as plt

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = scipy.io.loadmat('data/ex7data2.mat')
X = np.array(data['X'])

#%% Function Definition

def dist(x1, x2):
    return np.sqrt(np.sum(np.power((x1 - x2), 2)))

def findClosestCentroids(X, cent):
    return np.apply_along_axis(
        lambda x: np.argmin(np.array([dist(x, m) for m in cent])), 
        1, X)
    
def computeCentroids(X, idx, K):
    res = np.zeros((K, np.size(X,1)))
    for v in range(K):
        rand = np.mean(
                np.array([X[i] for i in range(len(X)) if idx[i] == v]), 
                axis=0)
        res[v] = rand
    return res
   
def runKMeans(X, init_cent, max_iter, isPlot=False):
    K = np.size(init_cent, 0)
    cent = init_cent
    prev_cent = cent
    if isPlot:
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(X[:,0], X[:,1], 'bo')
        ax.plot(init_cent[:,0], init_cent[:,1], 'go')     
        for i in range(max_iter):     
            idx = findClosestCentroids(X, cent)  
            cent = computeCentroids(X, idx, K) 
            if isPlot:
                ax.plot(cent[:,0], cent[:,1], 'rx')
                ax.plot(prev_cent[:,0], prev_cent[:,1], 'bx')
                for j in range(np.size(cent, 0)):
                    ax.arrow(prev_cent[j,0], prev_cent[j,1], 
                             cent[j,0] - prev_cent[j,0], 
                              cent[j,1] - prev_cent[j,1], 
                              head_width=0.05, head_length=0.1)   
        prev_cent = cent
        ax.plot(cent[:,0], cent[:,1], 'ro')      
    else:    
        for i in range(max_iter):     
            idx = findClosestCentroids(X, cent)  
            cent = computeCentroids(X, idx, K) 
    plt.show()
    return (cent, idx)

def randInitCentroids(X, K):
    return X[np.random.choice(X.shape[0], K)]

#%% Find Closest Centroids

print("Finding closest centriods. \n")

init_centroids = np.array([[3, 3], [6, 2], [8, 5]])

K = 3
idx = findClosestCentroids(X, init_centroids)

print("Closest centroids for the first 3 examples : ", 
      [x+1 for x in idx[:3]] ,"\n")
print("(The closest centroids should be 1, 3, 2 respectively) \n")

input("Program Paused. Press enter to continue. \n")

#%% Compute Means

print("Computing centroids mean. \n")
centroids = computeCentroids(X, idx, K)

print("Centroids computed after initial finding of closest centroids: \n")
print(centroids, " \n")
print("""(The centroids should be: \n
     [[ 2.428301 3.157924]
      [ 5.813503 2.633656] 
      [ 7.119387 3.616684]] )""")

input("Program Paused. Press enter to continue. \n")
   
#%% K-Means Clustering

print("Running K-Means clustering on example dataset. \n")

K = 3
max_iter = 10

init_centroids = np.array([[0, 0], [6, 0], [0, 6]])
cent, idx = runKMeans(X, init_centroids, max_iter, True)

print("K-Means Done. \n")

input("Program Paused. Press enter to continue. \n")

#%%  K-Means Clustering on Pixels: Display Image

A = plt.imread("data/bird_small.png")
plt.imshow(A)
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Image Compression

print("Applying K-Means to compress an image. \n")

origin = np.shape(A)

X = A.reshape((origin[0]*origin[1], 3))
K = 16
max_iter = 10
init_centroids = randInitCentroids(X, K)

cent, idx = runKMeans(X, init_centroids, max_iter, False)

idx = findClosestCentroids(X, cent)
X_recovered = cent[idx,:]
X_recovered = X_recovered.reshape(origin)

f, (ax1, ax2) = plt.subplots(2)
ax1.set_xlim([0, 128])
ax2.set_xlim([0, 128])
ax1.imshow(A)
ax2.imshow(X_recovered)
