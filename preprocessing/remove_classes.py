# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 10:15:05 2023

@author: Cornelius
"""
import cv2 as cv
from Display_mask import load_coco, find_image
from crop_from_mask import crop_from_mask, fill_mask
from simple_object_placer import coco_new_bbox
import matplotlib.pyplot as plt
from watershed_2_coco import export_json


def remove_by_class_id(dataset, class_id, PATH):
    removal = []
    for annotations in dataset["annotations"]:
        if annotations["category_id"] == class_id:
            image_name,image_id = find_image(dataset, annotations["id"]-1)
            cropped_im = fill_mask(dataset,image_id,annotations["segmentation"],image_name,PATH)
            cropped = crop_from_mask(dataset,annotations["id"]-1,cropped_im)
            plt.imshow(cropped)
            plt.axis("off")
            plt.show()
            bbox = coco_new_bbox(min(annotations["segmentation"][0][0::2]),min(annotations["segmentation"][0][1::2]),dataset,image_id,annotations["id"]-1)
            x = int(bbox[0])
            y = int(bbox[1])
            img = cv.imread(PATH + image_name)
            plt.imshow(img[y:y+int(bbox[3]), x:x+int(bbox[2])])
            temp_img = img[y:y+int(bbox[3]), x:x+int(bbox[2])] # Selecting a window of the image to edit
            temp_img[cropped>0] = 0 # All the places where the object is, is set to 0. Where the mask is 0, does remains unchanged from the larger_image
            img[y:y+cropped.shape[0], x:x+cropped.shape[1]] = temp_img # The window is put back into larger_image
            cv.imwrite("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/images/corrected/" + image_name,img)
            print(image_name)
            removal.append(annotations["id"]-1)
    offset = 0
    for idx in removal:
        print(dataset["annotations"][idx-offset]["category_id"])
        del dataset["annotations"][idx-offset]
        offset += 1
    return dataset
    
if __name__=="__main__":
    annotation_path = 'C:/Users/Cornelius/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/COCO_Training.json'
    #image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    image_dir = 'C:/Users/Cornelius/Downloads/DreierHSI_Mar_03_2023_09_18_Ole-Christian Galbo/Training/images/'
    dataset = load_coco(annotation_path)
    class_id = 1412700
    temp = remove_by_class_id(dataset, class_id,image_dir)
    export_json(temp,"COCO_corrected.json")
    