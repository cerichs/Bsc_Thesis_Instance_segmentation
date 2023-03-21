# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:06:01 2023

@author: admin
"""



import os
os.chdir(r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing")

from Display_mask import load_coco, load_annotation, find_image, draw_img
from crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
from watershed_2_coco import empty_dict, export_json
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

    

def extract_subwindow(original_img, new_annotation, window_size, img_id, image_dir, dataset):
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
                   

                    new_annotation["annotations"].append({'id': ann['id'],
                                           'image_id': ann['image_id'],
                                           'segmentation': [new_coords_coords],
                                           'iscrowd': ann['iscrowd'],
                                           'bbox': cropped_bbox_bbox,
                                           'area': ann['area'],
                                           'category_id': ann['category_id']})
                   
                    rr, cc = polygon(new_coords[1::2], new_coords[::2], shape=(window_height, window_width))
                    mask[rr, cc] = 1
                
                # plot every grain kernels contour
                #plt.plot(new_coords_coords[0::2], new_coords_coords[1::2], "o")
                #plt.show()
                
                """
                for mask in range(len(new_coords)):
                    for x_idx, y_idx in zip(range(len(new_coords[mask][0::2])), range(len(new_coords[mask][0::2]))):
                        
                        x = new_coords[mask][0::2][x_idx]
                        y = new_coords[mask][1::2][y_idx]
                        
                        legal = [(x+1, y),(x-1, y),(x, y+1),(x, y-1),(x+1, y+1),(x+1, y-1), (x-1, y+1),(x-1, y-1)]
                        
                        if (new_coords[mask][0::2][x_idx+1], new_coords[mask][1::2][y_idx+1]) in legal:
                            continue
                        
                        else:
                            corig = np.argmin()
                            new_coords.extend([new_coords[mask][0::2][x_idx+1], new_y])
                    
                            
                        #If the x's touches the edge it should fill in the gap 
                        # i.e. if the x+1 coordinate is >1 larger than x than it is a edge 
                        # and it should fill in
                        if abs(new_coords[mask][0::2][x+1] - new_coords[mask][0::2][x]) > 1:
                            
                            print(range(abs(new_coords[mask][0::2][x+1] - new_coords[mask][0::2][x])))
                        else:
                            continue
                        
                """
                #ann["bbox"][0] = ann["bbox"][0] - top_left_x
                #ann["bbox"][1] = ann["bbox"][1] - top_left_y 


                """
                if len(new_coords) > 0:
    
                    min_x, min_y = min(new_coords[::2]), min(new_coords[1::2])
                    max_x, max_y = max(new_coords[::2]), max(new_coords[1::2])
                    cropped_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                """
                    

    #print(ann["segmentation"])


   
    

    return subwindow, new_annotation
    #return subwindow, new_annotation, mask



################## MAIN ##################

#annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
annotation_path = r"C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Training\COCO_Training.json"
#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
image_dir = r'C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Training\images/'


new_annotation = empty_dict()
dataset = load_coco(annotation_path)
new_annotation["categories"] = dataset["categories"]


for c in range(1000):
    
    image_id = np.random.randint(0, len(dataset["images"]))   ### choose random image
    
    image_name = dataset["images"][image_id]["file_name"]
    image_id = image_id + 1
    
    # Get image-info from JSON
    image_path = os.path.join(image_dir, image_name)
    
    
    #BGR to RGB
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    
    subwindow_size = (256, 256)
    
    subwindow, new_annotation = extract_subwindow(img, new_annotation, subwindow_size, image_id, image_dir, dataset)
    #subwindow, new_annotation, mask = extract_subwindow(img, subwindow_size, image_id, image_dir, dataset)
    
    #c = image_id
    
    subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
    cv.imwrite(f"images/window{c}.jpg",subwindow)
    
    new_annotation['images'].append({'id':c,
                            'file_name': f"window{c}.jpg",
                            'license':1,
                            'height':subwindow.shape[0],
                            'width':subwindow.shape[1]})
    
export_json(new_annotation)
    
    
if False: 
    image_dir = "C:/Users/admin/Desktop/bachelor/Bsc_Thesis_Instance_segmentation/preprocessing/"
    
    annote_ids = []
    for i in range(len(new_annotation['annotations'])):
        if new_annotation['annotations'][i]['image_id']==image_id:
            annote_ids.append(i)
    c = image_id
    
    subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
    cv.imwrite(f"window{c}.jpg",subwindow)
        
    draw_img(new_annotation,image_id,annote_ids, image_dir)



"""
for c in range(30000):
    background = np.zeros((128,128,3),dtype = np.uint8)
    background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
    max_tries=100
    j=0
    while(j<max_tries):
        annotation_numb = np.random.randint(0,len(dataset['annotations']))
        #annotation_numb = 3792
        #if annotation_numb == 4485:
        #    print("bye bye error")
        image_name, image_id = find_image(dataset, annotation_numb)
        bbox, annotation = load_annotation(dataset, annotation_numb, image_id)
        cropped_im = fill_mask(dataset,image_id, annotation, image_name,image_dir)
        height, width = cropped_im.shape[0], cropped_im.shape[1]
        
        
        
        cropped = crop_from_mask(dataset, annotation_numb, cropped_im)
        x, y, keep = find_x_y(background, cropped,annotation, height,width)
        if keep:
            background = overlay_on_larger_image(background,cropped,x,y)
            dict_coco['annotations'].append({'id':coco_next_anno_id(dict_coco),
                                  'image_id':coco_next_img_id(dict_coco),
                                  'segmentation': [coco_new_anno_coords(dataset,image_id,annotation_numb,x,y)],
                                  'iscrowd':0,
                                  'bbox': coco_new_bbox(x,y,dataset,image_id,annotation_numb), #mangler x,y
                                  'area':dataset['annotations'][annotation_numb]['area'],
                                  'category_id':dataset['annotations'][annotation_numb]['category_id']
                                  })
        else:
            pass
        j+=1
    background= cv.cvtColor(background, cv.COLOR_BGR2RGB)
    cv.imwrite(f"images/Training/Synthetic_{c}.jpg",background)
    dict_coco['images'].append({'id':c+1,
                                'file_name': f"Synthetic_{c}.jpg",
                                'license':1,
                                'height':background.shape[0],
                                'width':background.shape[1]})
export_json(dict_coco)
# =============================================================================
# ov = np.zeros((300, 300))
# new_obj = np.zeros((300, 300))
# new_obj[10:20, 5:25] = 1
# 
# for i in range(700):
#     x, y = np.random.randint(0,300), np.random.randint(0,300)
#     if np.sum(ov[x:x+10,y:y+20])==0:
#         ov[x:x+10,y:y+20]=1
#     else:
#         continue
# 
# plt.imshow(ov)
# plt.show()
# =============================================================================
"""
