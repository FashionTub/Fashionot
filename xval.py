# -*- coding: utf-8 -*-
"""
Created on Tue May  8 21:09:44 2018

@author: Doug
"""


import os
import numpy as np
from skimage import io
import matrixgenerator as mg
import forwardprop as fp

# Root directory 
path = os.path.dirname(os.path.realpath('xval.py'))

# Folder paths
path_theta = path+r'\output'
folder1 = path+r'\images_transformed\fashionable'
folder2 = path+r'\images_transformed\unfashionable'

# Cross validation
X,Y = mg.generateMatrices('xval_set.txt','xval_set.txt')
X = X.astype(np.float64)
Y = Y.astype(np.float64)
m = X.shape[0]
n = X.shape[1]

# Neural Net Parameters
n2 = 100                #number of neurons in hidden layer 
n3 = 1                  #number of neurons in output layer
lam = 1
opt_theta = np.load(os.path.join(path_theta,'optimized_thetas.npy'))


#unpack the thetas
tTheta1 = opt_theta[0:n2*(n+1)]
tTheta2 = opt_theta[n2*(n+1):len(opt_theta)]
#reshape Theta1
tTheta1 = tTheta1.reshape((n2,n+1)) #Theta1 is n2 (hidden layer neurons) by n(input pixels)
#reshape Theta2
tTheta2 = tTheta2.reshape((n3,n2+1)) #Theta2 is n3 (output columns) by n2 (hidden layer neurons)
    
a3=fp.forwardprop(X, tTheta1, tTheta2)
acc = sum(Y==a3)/X.shape[0]*100
a = np.array2string(acc, precision = 2)+'%'
print ('Accuracy: ' + a.replace('[','').replace(']',''))
