# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:28:22 2023

@author: Cornelius
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from Display_mask import load_coco, load_annotation, find_image

def random_loc(xlim,ylim):
    np.random.seed(1)
    print(xlim,ylim)
    x,y = np.random.randint(0,high=xlim),np.random.randint(0,high=ylim)
    return x,y

def extract_img_from_mask(image,segmentation,mini_img):
    
    x, y = segmentation[0][0::2],segmentation[0][1::2] # comes in pair of [x,y,x,y,x,y], there split with even and uneven
    mini_img[]=True
    image = image[mini_img]
    return image

def placement(bbox,segmentation,image):
    xlim, ylim = bbox[2],bbox[3] # start at (0,0) and end at bbox width and height
    x, y = random_loc(xlim,ylim)
    mini_img = np.zeros((np.int8(xlim),np.int8(ylim),3),dtype=bool)
    segmentation[0][0::2] = [value-bbox[0] for value in segmentation[0][0::2]] #Make masks start at (0,0)
    segmentation[0][1::2] = [value-bbox[1] for value in segmentation[0][1::2]] #Make masks start at (0,0)
    new_img = extract_img_from_mask(image,segmentation,mini_img)
    return new_img, segmentation


annotation_path = 'C:/Users/Corne/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/COCO_training.json'
image_dir = 'C:/Users/Corne/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/images'
image_numb = 1
dataset = load_coco(annotation_path)
bbox, annotation = load_annotation(dataset, image_numb)
image_name = find_image(dataset,image_numb)
img = plt.imread(image_dir+image_name)
new_img, segmentation = placement(bbox,annotation,img)
plt.imshow(new_img)
plt.show()