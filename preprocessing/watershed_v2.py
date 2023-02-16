# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:32:07 2023

@author: Corne
"""

import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from watershed_2_coco import coco_dict, export_json

    
img_name = "test_img.PNG"
im = cv.imread(img_name) # Load image

im = cv.cvtColor(im,cv.COLOR_BGR2RGB) # Load image
grayscale = cv.cvtColor(im,cv.COLOR_RGB2GRAY) # Convert to Grayscale

auto_tresh = threshold_otsu(grayscale) # Determine Otsu threshold

segm_otsu = (grayscale > auto_tresh) # Apply threshold
img = segm_otsu.astype(int)*255 # Convert Bool type to 0:255 int


# Finding foreground area
# Finds the smallest distance between object and background
dist_transform = cv.distanceTransform(np.uint8(img),cv.DIST_L2,5)

l_max = peak_local_max(dist_transform, indices=False, min_distance=11,labels=img)
fg = np.int8(l_max)

# Marker labelling
ret, markers = cv.connectedComponentsWithAlgorithm(fg,connectivity=8,ccltype=cv.CCL_DEFAULT,ltype=cv.CV_32S )

markers = watershed(-dist_transform,markers,mask=img)
plt.imshow(markers)
plt.show()

fig, ax = plt.subplots(figsize=(10,12))
ax.imshow(im)
for cnt in np.unique(markers[markers > 0]):
    temp = markers.copy()
    temp[temp != cnt] = 0
    temp[temp == cnt] = 255
    
    #gray = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
    contours, hierarchy = cv.findContours(np.uint8(temp), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    new_array = np.squeeze(contours)
    temp=[]
    for j in range(len(new_array)):
        temp.append(int(new_array[j,0]))
        temp.append(int(new_array[j,1]))
    ax.plot(temp[0::2],temp[1::2],linestyle="-",linewidth=.5)
    
ax.axis('off')
plt.savefig("test.png",dpi=400)
plt.show()

#test_dict = coco_dict(img_name,markers)

#export_json(test_dict)
#plt.imshow(im)
#plt.fill(x,y,alpha=.7,color='g')
#plt.show()

# =============================================================================
# fix,(ax1,ax2)=plt.subplots(1,2)
# im[markers == -1] = [255,0,0]
# ax1.imshow(im)
# ax1.axis('off')
# ax2.imshow(markers == -1)
# ax2.axis('off')
# plt.show()
# =============================================================================
