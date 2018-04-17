# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 13:34:22 2018

@author: user
"""
import os
import os.path as path
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))


#%% Initialization

data = scipy.io.loadmat('data/ex7data1.mat')
X = np.array(data['X'])

raw_mat = scipy.io.loadmat('data/ex7faces.mat')
faceX = raw_mat["X"]

#%% Function Definition

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    return ((X - mu) / sigma, mu, sigma)

def pca(X):
    covar = np.matmul(X.T, X) / len(X)
    return (np.linalg.svd(covar))
    
def projectData(X, U, K):
    Z = np.zeros((len(X), K))
    Ud =  U[:, 0:K]
    for i in range(len(X)):
        Z[i] = np.matmul(X[i], Ud)
    return Z

def recoverData(Z, U, K):
    X_rec = np.zeros((len(Z), len(U)))
    for i in range(len(Z)):
        X_rec[i] = np.matmul(Z[i], U[:, 0:K].T)
    return X_rec

def displayData(X):
    m = int(np.sqrt(X.shape[0]))
    n = int(np.sqrt(X.shape[1]))
    fig, ax = plt.subplots(m,m,sharex=True,sharey=True)
    for i in range(m):
        for j in range(m):
            num = i*m + j
            img = np.transpose(X[num].reshape(n, n))
            ax[i][j].imshow(img, cmap='gray')
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xticklabels([])
    plt.show()

#%% Plot Data

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(X[:,0], X[:,1], 'bo')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Principal Component Analysis

print("Running PCA on example dataset. \n")

X_norm, mu, sigma = featureNormalize(X)
U, S, V = pca(X_norm)

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(X[:,0], X[:,1], 'bo')
ax.plot(np.array([mu[0], (mu + 1.5 * S[0] * U[:,0].T)[0]]), 
        np.array([mu[1], (mu + 1.5 * S[0] * U[:,0].T)[1]]), 
        'ro-')
ax.plot(np.array([mu[0], (mu + 1.5 * S[1] * U[:,1].T)[0]]), 
        np.array([mu[1], (mu + 1.5 * S[1] * U[:,1].T)[1]]), 
        'ro-')
plt.show()

print("Top eigenvector U[:,0]: ", U[:,0] ,"\n")
print("(You should expect to see -0.707107 0.707107) \n")

input("Program Paused. Press enter to continue. \n")

#%% Dimension Reduction 

K = 1
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)

print("Projection of the first example: ", Z[0], "\n")
print("(This value should be about 1.481274) \n")

print("Approximation of the first example: ", X_rec[0], "\n")
print("(This value should be about -1.047419 -1.047419) \n")

input("Program Paused. Press enter to continue. \n")

#%% Visualizing Dimension Reduction

fig, ax = plt.subplots(figsize=(8,8))
ax.plot(X_norm[:,0], X_norm[:,1], 'bo')
ax.plot(X_rec[:,0], X_rec[:,1], 'rx')
for i in range(len(X_norm)):
    ax.plot(np.array([X_norm[i, 0], X_rec[i, 0]]),
            np.array([X_norm[i, 1], X_rec[i, 1]]), 
            'k--')
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Visualizing Faces

displayData(faceX[:25,])

#%% PCA on Face Data

X_norm, mu, sigma = featureNormalize(faceX)
U, S, V = pca(X_norm)

print("Displaying Top 36 eigenvectors ... \n")
displayData(U[:36])

input("Program Paused. Press enter to continue. \n")

#%% PCA on Face Data

K = 100
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)

print("Original Faces \n")
displayData(X_norm[:25,])
print("Recovered Faces \n")
displayData(X_rec[:25,])

