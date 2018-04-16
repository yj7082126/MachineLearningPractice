# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:03:31 2018

@author: user
"""

import os
import os.path as path
import numpy as np
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.io import loadmat
import nltk

os.chdir(path.dirname(path.dirname(path.abspath(__file__))))

#%% Initialization

file_contents = open("data/emailSample1.txt").read()

mat = loadmat('data/spamTrain.mat')
X = mat['X']
y = mat['y']

mat2 = loadmat('data/spamTest.mat')
Xtest = mat2['Xtest']
ytest = mat2['ytest']

pos = np.array([X[i] for i in range(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in range(X.shape[0]) if y[i] == 0])

print('Total # of training emails = ', X.shape[0])
print('Number of training span emails = ', pos.shape[0])
print('Number of training nonspam emails = ', neg.shape[0])  

input("Program Paused. Press enter to continue. \n")

#%% Function Definition
 
def getVocabDict(reverse=False):
    f = open("data/vocab.txt").read().split("\n")
    key = [int(w.split("\t")[0])-1 for w in f if w != '']
    val = [(w.split("\t")[1]) for w in f if w != '']
    if reverse:
        df = pd.DataFrame({'wordind': key}, index=val)
    else:
        df = pd.DataFrame({'word': val}, index=key)
    return df

def processEmail(email):
    email = email.lower()
    email = re.sub('<[^<>]+>', '', email)
    email = re.sub('(<|>)', '', email)
    email = re.sub('[0-9]+', 'number', email)
    email = re.sub('(http|https)://[^\s]*', 'httpaddr', email)
    email = re.sub('[^\s]+@[^\s]+', 'emailaddr', email)
    email = re.sub('[$]+', 'dollar', email)
    email = re.sub('[\n]', ' ', email)
    
    print(" \n======== Proccessed Email ========\n")
    print(email)
    print("\n")
    
    stemmer = nltk.stem.porter.PorterStemmer()
    tokens = re.split(
            '[ \@\$\/\#\.\-\:\&\*\+\=\[\]\?\!\(\)\{\}\,\'\"\>\_\<\;\%]', email)
    vocab = getVocabDict(True)
    
    tokenlist = [stemmer.stem(re.sub('[^a-z-A-Z0-9]', '', token)) 
            for token in tokens if len(token) > 0]
    tokenlist = [x for x in tokenlist if x != '']
    
    indexlist = [vocab.loc[token, 'wordind'] for token in tokenlist 
                 if token in vocab.index]
    
    result = np.zeros(len(vocab))
    result[indexlist] = 1
    return email, tokenlist, indexlist, result

#%% Train Linear SVM for Spam Classification
    
linear_svm = SVC(C=0.1, kernel='linear')
linear_svm.fit(X, y.flatten())

train_predictions = linear_svm.predict(X)
train_acc = 100 * float(sum(train_predictions == y.flatten()))/y.shape[0]

print('Training accuracy = %0.2f%% \n' % train_acc)

input("Program Paused. Press enter to continue. \n")

#%% Test Linear SVM for Spam Classification

test_predictions = linear_svm.predict(Xtest)
test_acc = 100 * float(sum(test_predictions == ytest.flatten()))/ytest.shape[0]

print('Test set accuracy = %0.2f%% \n' % test_acc)

input("Program Paused. Press enter to continue. \n")

#%% Top predictors of Spam

sorted_weights = np.sort(linear_svm.coef_,axis=None)[::-1]
sorted_indicies = np.argsort(linear_svm.coef_, axis=None)[::-1]
vocabList = getVocabDict()

print("Top predictors of spam: \n")
for i in range(15):
    print("" + str(i+1) + ". " + vocabList.loc[sorted_indicies[i], "word"] + 
          " (" + str(sorted_weights[i]) + ") \n")

input("Program Paused. Press enter to continue. \n")
    
#%% Test Spam messages
    
filename = 'emailSample1.txt'

email = open("data/" + filename).read()
email2, tokenlist, indexlist, result = processEmail(email)
prediction = linear_svm.predict(result.reshape(1, -1))[0]

if prediction == 1:
    print("File " + filename+ " is a spam mail \n")
else:
    print("File " + filename+ " is not a spam mail \n")
