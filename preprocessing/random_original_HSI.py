# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:01 2023

@author: admin
"""

import os
#os.chdir(r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing")

from Display_mask import load_coco, load_annotation, find_image, draw_img
from crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
from watershed_2_coco import empty_dict, export_json
from simple_object_placer import coco_next_anno_id
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.draw import polygon
from random_original import extract_subwindow
from pls_opti import spectral_test



def mask_in_window(mask, window_top_x,window_bottom_x, window_top_y, window_bottom_y):
    is_in = False
    for i in range(len(mask[0::2])) :
        if (window_top_x <= mask[0::2][i] <= window_bottom_x) and  (window_top_y <= mask[1::2][i] <= window_bottom_y):
            is_in = True
        else:
            continue
    return is_in

    

def extract_subwindow_HSI(original_img, new_annotation, new_id, window_size, img_id, image_dir, image_name, dataset, z=1234, plot_mask=False):
    window_height, window_width, window_channels = window_size
    
    np.random.seed(z)
    top_left_x = np.random.randint(0, original_img.shape[1] - window_width)
    top_left_y = np.random.randint(0, original_img.shape[0] - window_height)

    bottom_right_x = top_left_x + window_width
    bottom_right_y = top_left_y + window_height

    subwindow = original_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x,:]

    #new_annotation = empty_dict()
    mask = np.zeros((window_height, window_width, window_channels), dtype=np.float64)
    
    # Precompute file_name
    #filename_dict = {file["id"]: file["file_name"] for file in dataset["images"]}
    
    for k, ann in enumerate(dataset['annotations']):
        if ann['image_id'] == image_id:
            
            if mask_in_window(ann['segmentation'][0], top_left_x, bottom_right_x, top_left_y, bottom_right_y):
                
                new_coords = []
                dup_dict = {}
                for coord_x, coord_y in zip(ann['segmentation'][0][0::2], ann['segmentation'][0][1::2]):
                    new_x = coord_x - top_left_x
                    new_y = coord_y - top_left_y
                    
                    if (0 <= new_x <= window_width) and (0 <= new_y <= window_height):
                        new_coords.extend([new_x, new_y])
                    
                if len(new_coords) > 0:
    
                    min_x, min_y = min(new_coords[::2]), min(new_coords[1::2])
                    max_x, max_y = max(new_coords[::2]), max(new_coords[1::2])
                    cropped_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                    
                new_coords_coords = []
                for coord_x, coord_y in zip(ann['segmentation'][0][0::2], ann['segmentation'][0][1::2]):
                    new_x = coord_x - top_left_x
                    new_y = coord_y - top_left_y
                    #Do not extend mask outside the subwindow
                    #Wall-E-algorithm
                    #Fills in the mask from edge grains
                    if new_x <= 0:
                        new_x = 0
                    
                    elif new_x >= window_width:
                        new_x = window_width -1 
                    
                    if new_y <= 0:
                        new_y = 0
                    
                    elif new_y >= window_height:
                        new_y = window_height -1

                    dup_dict[(new_x,new_y)] = 0
                    
                for x, y in dup_dict.keys():
                    if  (min_x <= x <= max_x) and (min_y <= y <= max_y):
                        new_coords_coords.extend([x, y])
                    else:
                        continue
                    
                if len(new_coords_coords) > 0:
    
                    min_x, min_y = min(new_coords_coords[::2]), min(new_coords_coords[1::2])
                    max_x, max_y = max(new_coords_coords[::2]), max(new_coords_coords[1::2])
                    cropped_bbox_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                   

                    new_annotation["annotations"].append({'id': coco_next_anno_id(new_annotation),
                                           'image_id': new_id,
                                           'segmentation': [new_coords_coords],
                                           'iscrowd': ann['iscrowd'],
                                           'bbox': cropped_bbox_bbox,
                                           'area': ann['area'],
                                           'category_id': ann['category_id']})
        

    new_annotation['images'].append({'id':new_id,
                            'file_name': f'{new_id}_window_{image_name}.jpg',
                            'license':1,
                            'height':subwindow.shape[0],
                            'width':subwindow.shape[1]})
        
    return subwindow, new_annotation



if __name__ == "__main__":

    #annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
    annotation_path = 'C:/Users/Corne/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/COCO_Test_orig.json'
    #annotation_path = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\COCO_Training.json"
    rgb_image_dir = 'C:/Users/Corne/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/images/'
    HSI_image_dir = r"I:\HSI\test"
    
    
    #rgb_image_dir = r'C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\images/'
    
    
    HSI_new_annotation = empty_dict()
    dataset = load_coco(annotation_path)
    HSI_new_annotation["categories"] = dataset["categories"]
    
    rgb_new_annotation = empty_dict()
    rgb_new_annotation["categories"] = dataset["categories"]
    
    #class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    class_check = [0]*8 
    class_check_Dense = [0]*8 
    class_check_sparse = [0]*8 
    c = 0
    n = 100
    while (c < n):
        
        image_id = np.random.randint(0, len(dataset["images"]))   ### choose random image
        
        image_name = dataset["images"][image_id]["file_name"]
        image_id = image_id + 1
        
        ids = [class_list.index(names) for names in class_list if names in image_name]
        if (class_check[ids[0]]<= (n//len(class_list))):
            keep = False
            if "Dense" in image_name:
                if class_check_Dense[ids[0]] <= (n//(len(class_list)*2)):
                    class_check_Dense[ids[0]] += 1
                    keep = True
            elif "Sparse" in image_name:
                if class_check_sparse[ids[0]] <= (n//(len(class_list)*2)):
                    class_check_sparse[ids[0]] += 1
                    keep = True
            if keep:
                class_check[ids[0]] += 1
                # Get image-info from JSON
                HSI_image_path = os.path.join(HSI_image_dir, image_name)
                rgb_image_path = os.path.join(rgb_image_dir, image_name)
                if not HSI_image_path.endswith(".npy"):
                    HSI_image_path = HSI_image_path[:-3]+"npy"
                    
                if not rgb_image_path.endswith(".jpg"):
                    rgb_image_path = rgb_image_path[:-3]+"jpg"
                    
                    
                
                #HSI_img = spectral_test(HSI_image_path)
                HSI_img = np.load(HSI_image_path)
                image_name = image_name.split(".")[0]
                
                HSI_subwindow_size = (256, 256, 102)
                
                HSI_subwindow, HSI_new_annotation = extract_subwindow_HSI(HSI_img, HSI_new_annotation, c, HSI_subwindow_size, image_id, HSI_image_dir, image_name, dataset, c, plot_mask=False)
                
                #BGR to RGB
                img = cv.imread(rgb_image_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                
                
                rgb_subwindow_size = (256, 256)
                
                #print(image_name)
                rgb_subwindow, rgb_new_annotation = extract_subwindow(img, rgb_new_annotation, c, rgb_subwindow_size, image_id, rgb_image_dir, image_name, dataset, c, plot_mask=False)
                #print(rgb_new_annotation)
                #extract_subwindow(original_img, new_annotation, new_id, window_size, img_id, image_dir, dataset, plot_mask=False):
                #subwindow, new_annotation, mask = extract_subwindow(img, subwindow_size, image_id, image_dir, dataset)
                
                #c = image_id
                
                #subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
                HSI_output_path = r"I:\HSI\HSI_Test_orig" 
                np.save(HSI_output_path + "\\"+ f"{c}_window_{image_name}.npy",HSI_subwindow)
                
                #rgb_output_path = r"I:\HSI\RGB_Test" 
                #subwindow = cv.cvtColor(rgb_subwindow, cv.COLOR_RGB2BGR)
                #cv.imwrite(rgb_output_path + "\\"+ f"{c}_window_{image_name}.jpg", subwindow)
                #np.save(rgb_output_path + f"window{c}.jpg", rgb_subwindow)
                
                c += 1 
    export_json(HSI_new_annotation,r"I:\HSI\test\COCO_HSI_windowed_test.json")
    #export_json(rgb_new_annotation,r"I:\HSI\test\COCO_rgb_windowed_test_PLS_eval.json")
    
   
