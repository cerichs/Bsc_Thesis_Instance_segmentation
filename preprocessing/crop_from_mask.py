# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:10:49 2023

@author: Cornelius
"""
from Display_mask import load_coco, load_annotation, find_image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.draw import polygon

def crop_from_mask(bbox,cropped_im,dataset):
    bbox=np.int16(bbox)

    start_x = bbox[1]
    end_x = bbox[1]+bbox[3]

    start_y = bbox[0]
    end_y = bbox[0]+bbox[2]
    
    cropped = cropped_im[start_x:end_x,start_y:end_y]
    return cropped

def fill_mask(image_id,annotation,image_name):
    height = dataset['images']['id'==image_id]['height']
    width = dataset['images']['id'==image_id]['width']
    mini_img = np.zeros((height,width),dtype=bool)
    x, y = (annotation[0][0::2]),(annotation[0][1::2])
    for x_x,y_y in zip(x,y):
        x_x, y_y = int(x_x), int(y_y)
        mini_img[y_y,x_x]=True
    img=mini_img.astype(int)

    row, col = polygon(y, x, img.shape)
    img[row,col] = 1
    orig_im = cv.imread(image_name)
    orig_im = cv.cvtColor(orig_im, cv.COLOR_BGR2RGB)

    cropped_im = cv.bitwise_and(orig_im, orig_im, mask=np.uint8(img))
    return cropped_im

def overlay_on_larger_image(larger_image,smaller_image,x_offset=None,y_offset=None):
    if x_offset == None:
        x = np.random.randint(0,larger_image.shape[1])
    if y_offset == None:
        y = np.random.randint(0,larger_image.shape[0])
    temp = larger_image[y:y+smaller_image.shape[0], x:x+smaller_image.shape[1]]
    temp[cropped>0] = 0
    temp += cropped*(cropped > 0)
    larger_image[y:y+smaller_image.shape[0], x:x+smaller_image.shape[1]] = temp
    plt.imshow(larger_image)
    plt.show()

annotation_path = 'C:/Users/Cornelius/OneDrive/DTU/Bachelor/COCO_testt.json'
image_dir = 'C:/Users/Cornelius/OneDrive/DTU/Bachelor/'
image_numb = 1
dataset = load_coco(annotation_path)
background = cv.imread("sunset.jfif")
background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
image_name,image_id = find_image(dataset, image_numb)

annote_ids = []
for i in range(len(dataset['annotations'])):
    if dataset['annotations'][i]['image_id']==image_numb:
        annote_ids.append(i)
for idx in annote_ids:
    bbox, annotation = load_annotation(dataset, idx,image_numb)
    cropped_im = fill_mask(image_id,annotation,image_name)
    cropped = crop_from_mask(bbox,cropped_im,dataset)
    overlay_on_larger_image(background,cropped)
    plt.imshow(cropped)
    plt.show()

