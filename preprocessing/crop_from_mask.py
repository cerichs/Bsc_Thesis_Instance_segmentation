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

def crop_from_mask(bbox,cropped_im):
    """Crops the region of interest in the cropped_im image
    
    Parameters
    ----------
    bbox : list, [x,y,width,height]
        The bounding box from the COCO dataset.
    cropped_im : Array-like image
        The image from fill_mask, where the background is removed and only
        region of intereset is shown, rest is black.

    Returns
    -------
    cropped : Array-like image
        Returns the cropped image with only the region of interest of dimension
        x*y

    """
    bbox=np.int16(bbox)

    start_x = bbox[1]
    end_x = bbox[1]+bbox[3]

    start_y = bbox[0]
    end_y = bbox[0]+bbox[2]
    
    cropped = cropped_im[start_x:end_x,start_y:end_y]
    return cropped

def fill_mask(image_id,annotation,image_name):
    """ Takes the segmentations from COCO dataset and discards the background 
    (ie. the region that is not withing interest)

    Parameters
    ----------
    image_id : int
        The image_id from the COCO dataset
    annotation : list, [x,y,x,y...x,y]
        The segmentation mask obtained from the COCO dataset
    image_name : str
        The name of the image where the mask orignates from

    Returns
    -------
    cropped_im : Returns the cropped image, where the area of interest has
    been cropped out and put on a black background

    """
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
    """ Place an object from a mask upon an image that is larger or same size as the object image
    Parameters
    ----------
    larger_image : array-like
        The background image to be overlayed upon
    smaller_image : array-like
        The object to be placed on the larger_image
    x_offset : int. Optional
        Where to place the object in respect to origo on the x-axis
        Optional. The default is None.
    y_offset : int. Optional
        Where to place the object in respect to origo on the y-axis
        The default is None.

    Returns
    -------
    Displays the resulting image.

    """
    if x_offset == None:
        x = np.random.randint(0,(larger_image.shape[1]-smaller_image.shape[1]))
    if y_offset == None:
        y = np.random.randint(0,(larger_image.shape[0]-smaller_image.shape[0]))
    temp = larger_image[y:y+smaller_image.shape[0], x:x+smaller_image.shape[1]] # Selecting a window of the image to edit
    temp[cropped>0] = 0 # All the places where the object is, is set to 0. Where the mask is 0, does remains unchanged from the larger_image
    temp += cropped * (cropped > 0) #the object is added to the blackened image
    larger_image[y:y+smaller_image.shape[0], x:x+smaller_image.shape[1]] = temp # The window is put back into larger_image
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
    cropped = crop_from_mask(bbox,cropped_im)
    overlay_on_larger_image(background,cropped)
    plt.imshow(cropped)
    plt.show()

