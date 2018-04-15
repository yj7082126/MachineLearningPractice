# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 16:17:58 2018

@author: yj7082126
"""

import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = loadmat('data/ex4data1.mat')
X = data['X']
y = data['y']
y = (y-1) % 10

data2 = loadmat('data/ex4weights.mat')
Theta1 = data2['Theta1']
Theta2 = data2['Theta2']
nn_params = np.append(Theta1.flatten(), Theta2.flatten())

#%% Function Definition

def sigmoid(X, theta):
    h = X.dot(theta)
    sig = 1 / (1 + np.exp(-h))
    return sig

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
    return fig, ax

def nnCostF(nn_params, inputs, hiddens, nums, 
            X, y, lambd):
    [m, n] = X.shape

    Theta1 = np.reshape(nn_params[:(hiddens)*(inputs + 1)], 
                                  (hiddens, inputs + 1))
    
    Theta2 = np.reshape(nn_params[(hiddens)*(inputs+1):], 
                                  (nums, hiddens + 1))
    
    a1 = np.append(np.ones((m, 1)), X, axis=1)
    z2 = sigmoid(a1, np.transpose(Theta1))
    a2 = np.append(np.ones((z2.shape[0], 1)), z2, axis=1)
    z3 = sigmoid(a2, np.transpose(Theta2))
    h = z3
    
    #Y = np.eye(num_labels)[y.flatten(),:]
    Y = np.zeros((m, nums))
    for i in range(m):
        Y[i, y[i]] = 1
    
    j1 = np.diag(np.log(h) @ np.transpose(Y))
    j2 = np.diag(np.log(1-h) @ np.transpose(1-Y))
    J = (-1/m) * sum(j1+j2)
    
    t1 = np.sum(np.square(Theta1[:,1:]))
    t2 = np.sum(np.square(Theta2[:,1:]))
    T = (lambd/(2*m)) * (t1+t2)
    
    J = J + T;
    
    D2 = 0;
    D3 = 0;
    
    for t in range(m):
        a1 = np.reshape(X[t], (1, n))
        a1 = np.append(np.ones((1, 1)), a1, axis=1)
        a2 = sigmoid(a1, np.transpose(Theta1))
        a2 = np.append(np.ones((1, 1)), a2, axis=1)
        a3 = sigmoid(a2, np.transpose(Theta2))
        
        d3 = a3 - Y[t]
        gz2 = np.multiply(a2, 1-a2)
        d2 = np.multiply((d3 @ Theta2), gz2)
        d2 = d2[:,1:]
        
        D3 = D3 + (np.transpose(d3) @ a2)
        D2 = D2 + (np.transpose(d2) @ a1)
        
    Theta1_grad = (1/m) * D2
    Theta2_grad = (1/m) * D3
    
    Theta1_grad[:,1:] = Theta1_grad[:,1:] + (lambd/m) * Theta1[:,1:] 
    Theta2_grad[:,1:] = Theta2_grad[:,1:] + (lambd/m) * Theta2[:,1:] 
        
    grad = np.append(Theta1_grad.flatten(), Theta2_grad.flatten())
    return J, grad

def sigGrad(X):
    sig = 1 / (1 + np.exp(-X))
    return np.multiply(sig, 1-sig)

def randInitW(Lin, Lout, epsil):
    W = np.random.rand(Lout, Lin+1) * (2*epsil) - epsil
    return W

def debugInitW(Fin, Fout):
    num = Fout * (Fin+1)
    W = np.reshape(np.sin(range(1, num+1)), (Fout, Fin+1), order='F') / 10
    return W

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

def checkNNGrad(*lambd):
    if not lambd:
        lambd = 0
    else:
        lambd = lambd[0]
  
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    
    Theta1 = debugInitW(input_layer_size, hidden_layer_size)
    Theta2 = debugInitW(hidden_layer_size, num_labels)
    
    X = debugInitW(input_layer_size-1, m)
    y = [(i % num_labels) for i in range(m)]
    
    nn_params = np.append(Theta1.flatten(), Theta2.flatten())
    
    costF = lambda p: nnCostF(p, input_layer_size, hidden_layer_size, 
                              num_labels, X, y, lambd)
    
    [cost, grad] = costF(nn_params)
    numgrad = computeNG(costF, nn_params)
    
    print(numgrad)
    print(grad)
    print("The above two matrices should be very similar to each other. \n")
    
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad) 
    print("""If the backpropagation implementation is correct, 
          then the relative difference will be less than 1e-9: \n
          Relative difference: """ , diff)
    
def predict(Theta1, Theta2, X):
    [m, n] = X.shape    
    
    if np.ndim(X) == 1:
        X = X.reshape((-1,1))
    
    D1 = np.hstack((np.ones((m,1)),X))

    hidden_pred = sigmoid(D1, np.transpose(Theta1))
    hidden_pred = np.hstack((np.ones((m,1)),hidden_pred)) 

    output_pred = sigmoid(hidden_pred, np.transpose(Theta2))
    
    p = np.argmax(output_pred,axis=1)
    
    return p    
 
#%% Plot Data
    
print('Visualizing Data... \n')
rand_ind = np.random.randint(0,len(X),100)
sel = X[rand_ind]

grid, ax = displayData(sel)
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Settings
    
input_layer_size = X.shape[1]
hidden_layer_size = 25
num_labels = len(np.unique(y))
epsil = 0.12

#%% Feedforward & Cost Function w/o Regularization

print('Feedforward Using Neural Network \n')
print('Checking Cost Function w/o Regularization \n')
lambd = 0
J, _ = nnCostF(nn_params, input_layer_size, hidden_layer_size, num_labels, 
               X, y, lambd)
print("Lamda 0: J = ", J, " (should be about 0.287629) \n")

input("Program Paused. Press enter to continue. \n")

#%% Feedforward & Cost Function w/o Regularization

print('Checking Cost Function w/ Regularization \n')
lambd = 1
J, _ = nnCostF(nn_params, input_layer_size, hidden_layer_size, num_labels, 
               X, y, lambd)
print("Lamda 1: J = ", J, " (should be about 0.383770) \n")

input("Program Paused. Press enter to continue. \n")

#%% Sigmoid Gradient

print('Evaluating Sigmoid Gradient \n')
g = sigGrad(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ', g)

input("Program Paused. Press enter to continue. \n")

#%% Initializing Parameters

print('Initializing Neural Network Parameters \n')
init_Theta1 = randInitW(input_layer_size, hidden_layer_size, epsil)
init_Theta2 = randInitW(hidden_layer_size, num_labels, epsil)

init_nnParams = np.append(init_Theta1.flatten(), init_Theta2.flatten())

#%% Implement Backpropagation w/o Regularization 

print('Checking Backpropagation w/o Regularization \n')

checkNNGrad()

input("Program Paused. Press enter to continue. \n")

#%% Implement Backpropagation w/ Regularization 

print('Checking Backpropagation w/ Regularization \n')

lambd = 3
checkNNGrad(lambd)

debugJ, _ = nnCostF(nn_params, input_layer_size, hidden_layer_size, num_labels, 
                 X, y, lambd)

print("Cost at debugging parameters (w/ lambda = ", lambd, 
        "): ", debugJ, 
        "\n For lambda = 3, this value should be about 0.576 \n")

input("Program Paused. Press enter to continue. \n")

#%% Training NN

print('Training Neural Network')
lambd = 0

results = minimize(nnCostF, init_nnParams, args=(input_layer_size, 
                hidden_layer_size, num_labels, X, y, lambd), method='CG', 
                jac=True, options={'maxiter': 50, 'disp':True})

params = results['x']

theta1 = np.reshape(params[:(hidden_layer_size)*(input_layer_size+1)], 
                                  (hidden_layer_size, input_layer_size + 1))

theta2 = np.reshape(params[(hidden_layer_size)*(input_layer_size+1):], 
                                  (num_labels, hidden_layer_size + 1))

input("Program Paused. Press enter to continue. \n")

#%% Visualize Neural Network

print('Visualizing Neural Network')
grid, ax = displayData(theta1[:,1:])
plt.show()

input("Program Paused. Press enter to continue. \n")

#%% Prediction

predictions = predict(theta1, theta2, X)
accuracy = np.mean(y == predictions.reshape((-1, 1))) * 100

print("Training Accuracy with neural network: ", accuracy, "%")