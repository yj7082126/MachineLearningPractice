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

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

data = scipy.io.loadmat('data/ex8_movies.mat')
Y = np.array(data['Y'])
R = np.array(data['R'])

data2 = scipy.io.loadmat('data/ex8_movieParams.mat')
X = np.array(data2['X'])
Theta = np.array(data2['Theta'])

#%% Function Definition

#%%
