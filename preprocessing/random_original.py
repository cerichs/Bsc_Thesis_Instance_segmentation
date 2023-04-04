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



def mask_in_window(mask, window_top_x,window_bottom_x, window_top_y, window_bottom_y):
    is_in = False
    for i in range(len(mask[0::2])) :
        if (window_top_x <= mask[0::2][i] <= window_bottom_x) and  (window_top_y <= mask[1::2][i] <= window_bottom_y):
            is_in = True
        else:
            continue
    return is_in

    

def extract_subwindow(original_img, new_annotation, new_id, window_size, img_id, image_dir, dataset, plot_mask=False):
    window_height, window_width = window_size

    top_left_x = np.random.randint(0, original_img.shape[1] - window_width)
    top_left_y = np.random.randint(0, original_img.shape[0] - window_height)

    bottom_right_x = top_left_x + window_width
    bottom_right_y = top_left_y + window_height

    subwindow = original_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    #new_annotation = empty_dict()
    mask = np.zeros((window_height, window_width), dtype=np.uint8)

    for ann in dataset['annotations']:
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
                    
                    elif new_x > window_width:
                        new_x = window_width
                    
                    if new_y <= 0:
                        new_y = 0
                    
                    elif new_y > window_height:
                        new_y = window_height

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
                            'file_name': f"window{new_id}.jpg",
                            'license':1,
                            'height':subwindow.shape[0],
                            'width':subwindow.shape[1]})
    
    
    subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
    cv.imwrite(f"images/window{new_id}.jpg",subwindow)

    if plot_mask == True:
        annote_ids = []
        
        #print (new_annotation['annotations'] )
        for i in range(len(new_annotation['annotations'])):
            #print(i)
            if new_annotation['annotations'][i]['image_id']==new_id:
                #print(image_id)
                print(i)
                annote_ids.append(i)
            else:
                continue
        
        image_dir2 = r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\images/"
        draw_img(new_annotation,new_id, annote_ids, image_dir2)
        
    return subwindow, new_annotation



if __name__ == "__main__":

    #annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
    annotation_path = 'C:/Users/Cornelius/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/COCO_Training.json'
    image_dir = 'C:/Users/Cornelius/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/images/'
    
    
    new_annotation = empty_dict()
    dataset = load_coco(annotation_path)
    new_annotation["categories"] = dataset["categories"]
    
    #class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    class_check = [0]*8 
    class_check_Dense = [0]*8 
    class_check_sparse = [0]*8 
    c = 0
    n = 1000
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
                image_path = os.path.join(image_dir, image_name)
                
                
                #BGR to RGB
                img = cv.imread(image_path)
                #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                
                
                subwindow_size = (256, 256)
                
                subwindow, new_annotation = extract_subwindow(img, new_annotation, c, subwindow_size, image_id, image_dir, dataset, plot_mask=False)
                #subwindow, new_annotation, mask = extract_subwindow(img, subwindow_size, image_id, image_dir, dataset)
                
                #c = image_id
                
                subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
                cv.imwrite(f"images/window{c}.jpg",subwindow)
                c += 1
            
        
        ### Extracting name of the particular grain-type and counting instances in image
        ground_truth = 0
        name = []
        for annotations in new_annotation["annotations"]:
            break
            if annotations["image_id"]==c:
                ground_truth += 1
                for categories in new_annotation["categories"]:
                    if categories["id"] == annotations["category_id"]:
                        #print(annotations["category_id"])
                        name.append(categories["name"])
        #name = set(name)
        #print("")
        #print(f"The following grain-type being analysed is:  {name}   with image_id:  {image_id}")
        #print("")
        #print(f"The ground-truth amount of kernels in the image is:  {ground_truth}")
        
        
    export_json(new_annotation,"COCO_windowed_2k.json")
    
   
