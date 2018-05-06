# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 17:42:27 2018

@author: dzhang
"""

import os
import compress

# Root directory 
path = os.path.dirname(os.path.realpath('rateIMG.py'))

# Folder paths
folder_input = path+r'\input'
folder_input_transform = folder_input+r'\transform'

# Create folders
if not os.path.exists(folder_input):
    os.makedirs(folder_input)
    
if not os.path.exists(folder_input_transform):
    os.makedirs(folder_input_transform)

compress.compress_image(folder_input, folder_input_transform)
compress.convert_jpg(folder_input_transform)