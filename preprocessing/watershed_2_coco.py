# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:56:18 2023

@author: Cornelius
"""

import os
import json
import IPython
import cv2 as cv
import numpy as np

def empty_dict():
    dict_coco = {'annotations':[],
                 'categories':[],
                 'images':[],
                 'info':None,
                 'licenses':[]}
    return dict_coco

def coco_dict(filename,markers):
    temp_img = cv.imread(filename)
    height, width, channels = np.shape(temp_img)
    edge_img = np.uint8(markers == -1)
    contours, hierarchy = cv.findContours(edge_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE to avoid RLE

    dict_coco = empty_dict()
    for i in range(len(contours)):
        dict_coco['annotations'].append({'id':None,
                                      'image_id':None,
                                      'segmentation': [watershed_2_coco(contours[i])],
                                      'iscrowd':0,
                                      'bbox': [0,0,width, height], #mangler x,y
                                      'area':None,
                                      'category_id':None
                                      })
    return dict_coco

def watershed_2_coco(contours):
    new_array = np.squeeze(contours)
    temp=[]
    for j in range(len(new_array)):
        temp.append(int(new_array[j,0]))
        temp.append(int(new_array[j,1]))
        
    return temp

def export_json(dict_coco):
    with open('COCO_export.json', 'w') as fp:
        json.dump(dict_coco, fp)


annotation_path = 'C:/Users/Cornelius/OneDrive/DTU/Bachelor/COCO_testt.json'
#annotation_path = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/COCO_export.json'
json_open = open(annotation_path)
masks = json.load(json_open)
json_open.close()

