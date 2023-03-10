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
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.draw import polygon

def find_x_y(larger_image,smaller_image):
    run=True
    count = 0
    while(run):
        x = np.random.randint(0,(larger_image.shape[1]-smaller_image.shape[1]))
        y = np.random.randint(0,(larger_image.shape[0]-smaller_image.shape[0]))
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

annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
dataset = load_coco(annotation_path)
background = np.zeros((500,500,3),dtype = np.uint8)
background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
max_tries=2000
j=0
while(j<max_tries):
    annotation_numb = np.random.randint(0,len(dataset['annotations']))
    image_name,image_id = find_image(dataset, annotation_numb)
    #print(image_name)
    #print(j)
    bbox, annotation = load_annotation(dataset, annotation_numb,image_id)
    cropped_im = fill_mask(dataset,image_id,annotation,image_name)
    cropped = crop_from_mask(bbox,cropped_im)
    x, y, keep = find_x_y(background,cropped)
    if keep:
        background = overlay_on_larger_image(background,cropped,x,y)
    else:
        pass
    j+=1
plt.imshow(background)
plt.axis('off')
plt.savefig("test_image.jpg",dpi=300)
plt.show()


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

