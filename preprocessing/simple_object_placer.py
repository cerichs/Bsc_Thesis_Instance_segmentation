# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:04:37 2023

@author: Cornelius
"""
import numpy as np
import matplotlib.pyplot as plt
#from crop_from_mask import overlay_on_larger_image
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:10:49 2023

@author: Cornelius
"""
from Display_mask import load_coco, load_annotation, find_image
from crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
from watershed_2_coco import empty_dict, export_json
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.draw import polygon

def find_x_y(larger_image,smaller_image,annotation,height,width):
    run=True
    count = 0
    while(run):
        x = np.random.randint(0,(larger_image.shape[1]-smaller_image.shape[1]))
        y = np.random.randint(0,(larger_image.shape[0]-smaller_image.shape[0]))
        x,y = edge_grain(annotation,height,width,x,y,larger_image)
        temp = larger_image[y:y+smaller_image.shape[0], x:x+smaller_image.shape[1]]
        overlap_amount = np.sum(np.multiply(temp,smaller_image))
        #print(overlap_amount)
        if overlap_amount<25000: #TODO make it a ratio of kernel, as smaller kernels constantly overlapped
            run = False
            keep = True
        count+=1
        if count>100:
            run = False
            keep = False
    return x, y, keep

def edge_grain(annotation,height,width,x,y,larger_image):
    if (max(annotation[0][1::2])>=height-1):
        return x,(larger_image.shape[0]-(max(annotation[0][1::2])-min(annotation[0][1::2])))-1
    elif (max(annotation[0][0::2])>=width-1):
        return (larger_image.shape[1]-(max(annotation[0][0::2])-min(annotation[0][0::2])))-1,y
    elif (min(annotation[0][1::2])<=1):
        return x,1
    elif(min(annotation[0][0::2])<=1):
        return 1,y
    elif (max(annotation[0][1::2])>=height-1) and (max(annotation[0][0::2])>=width-1):
        return (larger_image.shape[1]-(max(annotation[0][0::2])-min(annotation[0][0::2])))-1, (larger_image.shape[0]-(max(annotation[0][1::2])-min(annotation[0][1::2])))-1
    elif (min(annotation[0][1::2])<=1) and (min(annotation[0][0::2])<=1):
        return 1,1
    else:
        return x,y

def coco_next_anno_id(coco_dict):
    return len(coco_dict['annotations'])+1

def coco_next_img_id(coco_dict):
    return len(coco_dict['images'])+1

def coco_new_anno_coords(dataset,image_id,annotation_numb,x,y):
    width = min(dataset['annotations'][annotation_numb]['segmentation'][0][0::2])
    height = min(dataset['annotations'][annotation_numb]['segmentation'][0][1::2])
    new_annote = [None]*len(dataset['annotations'][annotation_numb]['segmentation'][0]) # new list the same size
    new_annote[0::2] = [z - width for z in dataset['annotations'][annotation_numb]['segmentation'][0][0::2]] # rescale to start at 0
    new_annote[1::2] = [z - height for z in dataset['annotations'][annotation_numb]['segmentation'][0][1::2]] # rescale to start at 0
    new_annote[0::2] = [z + x for z in new_annote[0::2]] # rescale for new img
    new_annote[1::2] = [z + y for z in new_annote[1::2]] # rescale for new img
    return new_annote

def coco_new_bbox(x,y,dataset,image_id,annotation_numb):
    annote = coco_new_anno_coords(dataset,image_id,annotation_numb,x,y)
    width = max(annote[0::2])-min(annote[0::2])
    height = max(annote[1::2])-min(annote[1::2])
    return [x,y,width,height]

#annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
annotation_path = 'C:/Users/Corne/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/COCO_training.json'
#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
image_dir = 'C:/Users/Corne/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/images/'
dataset = load_coco(annotation_path)
dict_coco = empty_dict()
dict_coco['categories']=dataset['categories']
dict_coco['info']=dataset['info']
dict_coco['licenses']=dataset['licenses']
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

