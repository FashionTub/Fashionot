# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:42:27 2018

@author: dzhang
"""

import os
import numpy as np
from skimage import io
import compress as comp
#import matrixgenerator as mg
import forwardprop as fp

# Root directory 
path = os.path.dirname(os.path.realpath('rateIMG.py'))

# Folder paths
path_theta = path+r'\output'
folder_input = path+r'\input'
folder_input_transform = folder_input+r'\transform'

# Create folders
if not os.path.exists(folder_input):
    os.makedirs(folder_input)
    
if not os.path.exists(folder_input_transform):
    os.makedirs(folder_input_transform)

comp.compress_image(folder_input, folder_input_transform)
comp.convert_jpg(folder_input_transform)

# Neural Net Parameters
n2 = 100                #number of neurons in hidden layer 
n3 = 1                  #number of neurons in output layer
lam = 1
opt_theta = np.load(os.path.join(path_theta,'optimized_thetas.npy'))
    
### NEED TO FIX matrixgenerator so that it can be used here
### For now copy code for matrixgenerator     
# generate matrix for a single input
def generateMatrices(filepath):
    #give file dictionaries for folder containing transformed inputs 
    X = np.empty((0,30000), 'uint8')	   
    #for all images in folder 1  

    filelist = [f for f in os.listdir(filepath) if os.path.isfile(os.path.join(filepath, f))]  
    #filelist = [x.strip() for x in filelist] 
    
    for imagefilename in filelist:
      #load image      
      image = io.imread(os.path.join(filepath,imagefilename))
      #convert the data from 0-255 to -1 to 1
      image = (2/255)*image
      image = image -1
      #reshape image newshape
      datarow = np.reshape(image, (1,-1)) # make a 1 x m matrix where m is however many pixels
                                # there are in the image    
      X=np.append(X,datarow,axis=0)

    return X


def fashionot():
    X = generateMatrices(folder_input_transform)
    n = X.shape[1]
    
    #unpack the thetas
    tTheta1 = opt_theta[0:n2*(n+1)]
    tTheta2 = opt_theta[n2*(n+1):len(opt_theta)]
    #reshape Theta1
    tTheta1 = tTheta1.reshape((n2,n+1)) #Theta1 is n2 (hidden layer neurons) by n(input pixels)
    #reshape Theta2
    tTheta2 = tTheta2.reshape((n3,n2+1)) #Theta2 is n3 (output columns) by n2 (hidden layer neurons)  
    
    a3=fp.forwardprop(X, tTheta1, tTheta2)
    print(a3)
    print("Done")

fashionot()