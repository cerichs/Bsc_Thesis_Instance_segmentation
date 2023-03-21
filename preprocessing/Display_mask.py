# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:04:59 2023

@author: Cornelius
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
import os
import json
import IPython


def load_coco(path):
    json_open = open(path)
    masks = json.load(json_open)
    json_open.close()
    ## TODO: add to dict, feature that shows tracks all the annotations belonging to 1 image
    return masks

def load_annotation(dataset, annotation_numb,image_numb):
    temp = dataset['annotations']
    image_mask = temp[annotation_numb]
    bbox = image_mask['bbox']
    annotation = image_mask['segmentation']
    return bbox, annotation

def find_image(dataset,annotation_numb):
    image_id=dataset['annotations'][annotation_numb]['image_id']
    for i in range(len(dataset['images'])):
        if dataset['images'][i]['id']==image_id:
            image_name=dataset['images'][i]['file_name']
    
    return image_name,image_id

def draw_img(dataset,image_numb,annote_ids, image_dir):
    image_name,image_id = find_image(dataset,annote_ids[0])
    img = plt.imread(image_dir+image_name)
    fix,ax=plt.subplots()
    ax.imshow(img)
    for ids in annote_ids:
        bbox,annotation = load_annotation(dataset, ids, image_numb) # Get bounding box and annotations
        bbox_x, bbox_y, width, height = bbox
        ax.add_patch(plt.Rectangle((bbox_x, bbox_y), width, height,linewidth=1,edgecolor='b', facecolor='none'))
        x, y = annotation[0][0::2],annotation[0][1::2] # comes in pair of [x,y,x,y,x,y], there split with even and uneven
        plt.fill(x, y,alpha=.7)
    plt.show()
imported = True
if not imported:  
    annotation_path = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/COCO_export.json'
    image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    image_numb = 1
    dataset = load_coco(annotation_path)
    annote_ids = []
    for i in range(len(dataset['annotations'])):
        if dataset['annotations'][i]['image_id']==image_numb:
            annote_ids.append(i)
    draw_img(dataset,image_numb,annote_ids)