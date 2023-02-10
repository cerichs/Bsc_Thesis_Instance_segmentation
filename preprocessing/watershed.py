# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:30:53 2023

@author: Cornelius
"""

import numpy as np
from cv2 import watershed
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import color, io, measure, segmentation
from skimage.filters import threshold_otsu
from skimage.morphology import erosion,dilation, square
from tensorflow.keras.preprocessing import image


    
img_name = 'pixelwise.png'
im = cv.imread(img_name) # Load image
grayscale = cv.cvtColor(im,cv.COLOR_RGB2GRAY) # Convert to Grayscale

auto_tresh = threshold_otsu(grayscale) # Determine Otsu threshold

segm_otsu = (grayscale < auto_tresh) # Apply threshold
img = segm_otsu.astype(int)*255 # Convert Bool type to 0:255 int

# Morp opening, with 2 iterations, (erode->erode->dilate->dilate)
footprint = square(3) # 3x3 square kernel
eroded = erosion(img, footprint)
eroded = erosion(eroded, footprint)
dilated = dilation(eroded, footprint)
dilated = dilation(dilated, footprint)
img_dil = np.uint8(dilated) #Change back to int8 instead of int32


#Finding background with dilation 3 iterations
bg_1 = dilation(img_dil, footprint)
bg_2 = dilation(bg_1, footprint)
bg_3 = dilation(bg_2, footprint)


# Finding foreground area
dist_transform = cv.distanceTransform(img_dil,cv.DIST_L2,5)
ret, fg = cv.threshold(dist_transform,0.5*dist_transform.max(),255,0)
# Finding unknown region
fg = np.uint8(fg)

unknown = bg_3 - fg  #The unknown area is the area that is not the foreground or background

# Marker labelling
ret, markers = cv.connectedComponents(fg)

# Add one so background is 1
markers = markers+1
# Rest is unknown and is marked 0
markers[unknown==255] = 0

markers = watershed(im,markers)

markers = segmentation.clear_border(markers,bgval=1)


plt.imshow(markers == -1)
plt.show()