# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:24:04 2018

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
import seaborn as sns

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))


#%% Initialization

data = scipy.io.loadmat('data/ex8_movies.mat')
Y = np.array(data['Y'])
R = np.array(data['R'])

data2 = scipy.io.loadmat('data/ex8_movieParams.mat')
X = np.array(data2['X'])
Theta = np.array(data2['Theta'])
num_features = data2['num_features'].flatten()[0]
num_movies = data2['num_movies'].flatten()[0]
num_users = data2['num_users'].flatten()[0]

mList = [x.split(' ', 1)[1] for x in open("data/movie_ids.txt", encoding='latin-1').read().split("\n") if x != '']

#%% Function Definition

def cofiCostFunc(nn_params, Y, R, features, movies, users, lambd):
    
    X = np.reshape(nn_params[:(features)*(movies)], 
                                  (movies, features))
    
    Theta = np.reshape(nn_params[(features)*(movies):], 
                                  (users, features))
    
    P1 = X.dot(np.transpose(Theta)) - Y
    P2 = np.square(np.multiply(P1, R))
    J = (1/2) * P2.sum()
    
    X_grad = np.multiply(P1, R).dot(Theta)
    Theta_grad = np.transpose(np.multiply(P1, R)).dot(X)
    
    Jreg = (lambd/2) * (np.square(Theta).sum() + np.square(X).sum())
    J = J + Jreg
    
    Xreg = lambd * X
    Thetareg = lambd * Theta
    X_grad += Xreg
    Theta_grad += Thetareg
    
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())
    return J, grad

def computeNG(J, theta):
    ngrad = np.zeros((theta.shape))
    pturb = np.zeros((theta.shape))
    e = 10 ** -4
    for p in range(len(theta)):
        pturb[p] = e
        l1, _ = J(theta - pturb)
        l2, _ = J(theta + pturb)
        ngrad[p] = (l2 - l1) / (2*e)
        pturb[p] = 0
    return ngrad

def checkCostF(*lambd):
    if not lambd:
        lambd = 0
    else:
        lambd = lambd[0]
  
    features = 3
    movies = 5
    users = 4
    
    X_t = np.random.rand(movies, features)
    Theta_t = np.random.rand(users, features)
    
    Y = X_t.dot(np.transpose(Theta_t))
    Y_ind = (np.random.rand(movies, users) > 0.5)
    Y[Y_ind] = 0
    R = np.zeros((movies, users))
    R[Y != 0] = 1
    
    X = np.random.normal(0, 1, (movies, features))
    Theta = np.random.normal(0, 1, (users, features))
    
    nn_params = np.append(X.flatten(), Theta.flatten())
    
    cCost = lambda p: cofiCostFunc(p, Y, R, features, movies, users, lambd)
    
    [cost, grad] = cCost(nn_params)
    numgrad = computeNG(cCost, nn_params)
    
    print(numgrad)
    print(grad)
    print("The above two matrices should be very similar to each other. \n")
    
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad) 
    print("""If the backpropagation implementation is correct, 
          then the relative difference will be less than 1e-9: \n
          Relative difference: """ , diff)
    
def normalizeRatings(Y, R):
    Ymean = np.apply_along_axis(lambda x: np.mean([y for y in x if y != 0]), 1, Y)
    Ynorm = Y - Ymean.reshape(-1, 1)
    Ynorm[np.where(R == 0)] = 0
    return Ymean, Ynorm
        
#%% Plot Data
toystory = Y[0,np.argwhere(R[0]).flatten()].mean()
print("Average rating for movie 1 (Toy Story): ", toystory, "\n")

fig, ax = plt.subplots()
sns.heatmap(Y, ax=ax, annot=False)

plt.xlabel("Users")
plt.ylabel("Movies")
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Collaborative Filtering Cost Function (w/o lambda)

tmp_features = 3
tmp_movies = 5
tmp_users = 4

tmp_X = X[0:tmp_movies, 0:tmp_features]
tmp_Theta = Theta[0:tmp_users, 0:tmp_features]
tmp_Y = Y[0:tmp_movies, 0:tmp_users]
tmp_R = R[0:tmp_movies, 0:tmp_users]

nn_params = np.append(tmp_X.flatten(), tmp_Theta.flatten())
lambd = 0

J, _ = cofiCostFunc(nn_params, tmp_Y, tmp_R, tmp_features, 
                                     tmp_movies, tmp_users, lambd)

print("Cost at loaded parameters (lambd = 0): ", J, "(should be about 22.22) \n")

input("Program Paused. Press enter to continue. \n")

#%% Collaborative Filtering Gradient (w/o lambda)

print("Checking Gradients (w/o regularization) ... \n")

checkCostF()

input("Program Paused. Press enter to continue. \n")

#%% Collaborative Filtering Cost Function (w/ lambda)

lambd = 1.5
J, _ = cofiCostFunc(nn_params, tmp_Y, tmp_R, tmp_features, 
                                     tmp_movies, tmp_users, lambd)


print("Cost at loaded parameters (lambd = 1.5): ", J, "(should be about 31.34) \n")

input("Program Paused. Press enter to continue. \n")

#%% Collaborative Filtering Gradient (w/o lambda)

print("Checking Gradients (w/ regularization) ... \n")

checkCostF(1.5)

input("Program Paused. Press enter to continue. \n")

#%% Entering ratings for a new user

myRatings = np.zeros(len(mList))

myRatings[0] = 4
myRatings[6] = 3
myRatings[11]= 5
myRatings[53]= 4
myRatings[63]= 5
myRatings[65]= 3
myRatings[68]= 5
myRatings[97] = 2
myRatings[182]= 4
myRatings[225]= 5
myRatings[354]= 5

print("New User Ratings ... \n")
for x in range(len(mList)):
    if myRatings[x] > 0:
        print("Rated", myRatings[x], "for", mList[x], " \n")
        
input("Program Paused. Press enter to continue. \n")

#%% learning Movie Ratings

print("Training collaborative filtering... \n")
num_users += 1

Y = np.hstack((myRatings.reshape(-1, 1), Y))
R = np.hstack(((myRatings.reshape(-1, 1) != 0)*1, R))

Ymean, Ynorm = normalizeRatings(Y, R)

X = np.random.rand(num_movies, num_features)
Theta = np.random.rand(num_users, num_features)

nn_params = np.append(X.flatten(), Theta.flatten())
lambd = 10

cCost = lambda p: cofiCostFunc(p, Ynorm, R, num_features, num_movies, num_users, lambd)

results = minimize(cCost, nn_params, method='CG', jac=True, options={'maxiter':100, 'disp':True})['x']

X = np.reshape(results[:(num_features)*(num_movies)], 
                              (num_movies, num_features))

Theta = np.reshape(results[(num_features)*(num_movies):], 
                              (num_users, num_features))

input("Program Paused. Press enter to continue. \n")

#%% Recommendation

p = X.dot(np.transpose(Theta))
pred = p[:,0] + Ymean
sortpred = np.sort(pred)
sortidx = pred.argsort()[::-1]

print("Top Recommendations for you: \n")
for i in sortidx[:10]:
    print("Predicting rating ", round(pred[i]), " for movie ", mList[i], "\n")