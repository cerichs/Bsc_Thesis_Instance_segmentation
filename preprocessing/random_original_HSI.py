import sys
sys.path.append("..")

from .Display_mask import load_coco
from two_stage.watershed_2_coco import empty_dict, export_json
from .simple_object_placer import coco_next_anno_id
from .random_original import extract_subwindow
from .preprocess_image import spectral_test

import numpy as np
import cv2 as cv
import os


def mask_in_window(mask, window_top_x,window_bottom_x, window_top_y, window_bottom_y):
    """
    Checks if mask coordinates are within the given window.
    
    Args:
        mask (list): List of mask coordinates.
        window_top_x (int): Top-left x-coordinate of the window.
        window_bottom_x (int): Bottom-right x-coordinate of the window.
        window_top_y (int): Top-left y-coordinate of the window.
        window_bottom_y (int): Bottom-right y-coordinate of the window.
    
    Returns:
        bool: True if mask is in the window, False otherwise.
    """
    is_in = False
    for i in range(len(mask[0::2])) :
        if (window_top_x < mask[0::2][i] < window_bottom_x) and  (window_top_y < mask[1::2][i] < window_bottom_y):
            is_in = True
        else:
            continue
    return is_in



def extract_subwindow_HSI(original_img, new_annotation, new_id, window_size, image_id, image_dir, image_name, dataset, z=1234, plot_mask=False):
    """
    Extract a subwindow from the original image and update the annotation information.

    Args:
        original_img (ndarray): Original image.
        new_annotation (dict): New annotation dictionary.
        new_id (int): New ID for the extracted subwindow.
        window_size (tuple): Window size as (height, width).
        image_id (int): Image ID of the original image.
        image_dir (str): Directory containing the original image.
        dataset (dict): Original dataset information.
        z (int, optional): Random seed. Defaults to 1234.
        plot_mask (bool, optional): Whether to plot the mask. Defaults to False.

    Returns:
        tuple: Tuple containing the extracted subwindow (ndarray) and the updated new_annotation (dict).
    """
    
    # Get window dimensions
    window_height, window_width, window_channels = window_size
    
    # Set random seed for reproducibility
    np.random.seed(z)
    
    # Generate random top left corner for the subwindow
    top_left_x = np.random.randint(0, original_img.shape[1] - window_width)
    top_left_y = np.random.randint(0, original_img.shape[0] - window_height)

    # Calculate bottom right corner of the subwindow
    bottom_right_x = top_left_x + window_width
    bottom_right_y = top_left_y + window_height

    # Extract the subwindow
    subwindow = original_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # Initialize a new mask for the subwindow
    mask = np.zeros((window_height, window_width), dtype=np.uint8)
    
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
                    
                    if (0 < new_x < window_width-1 ) and (0 < new_y < window_height-1):
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
                        new_x = window_width - 1
                    
                    if new_y <= 0:
                        new_y = 0
                    
                    elif new_y >= window_height:
                        new_y = window_height - 1

                    dup_dict[(new_x,new_y)] = 0
                    
                for x, y in dup_dict.keys():
                    if 'min_x' in locals() and 'min_y' in locals() and 'max_x' in locals() and 'max_y' in locals():
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




def extract_subwindow_main(annotation_path, HSI_image_dir, rgb_image_dir, n=100):

    # Initialize empty annotation dictionaries for HSI and RGB images
    HSI_new_annotation = empty_dict()
    dataset = load_coco(annotation_path)
    HSI_new_annotation["categories"] = dataset["categories"]

    rgb_new_annotation = empty_dict()
    rgb_new_annotation["categories"] = dataset["categories"]

    # Initialize other variables
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    class_check = [0]*8 
    class_check_Dense = [0]*8 
    class_check_sparse = [0]*8 
    c = 0

    while (c < n):

        image_id = np.random.randint(0, len(dataset["images"]))   ### choose random image
        
        image_name = dataset["images"][image_id]["file_name"]
        image_id = image_id + 1
        
        ids = [class_list.index(names) for names in class_list if names in image_name]
        if (class_check[ids[0]] <= (n//len(class_list))):
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
                    
                HSI_img = spectral_test(HSI_image_path)
                
                image_name = image_name.split(".")[0]
                
                HSI_subwindow_size = (256, 256, 102)
                
                HSI_subwindow, HSI_new_annotation = extract_subwindow_HSI(HSI_img, HSI_new_annotation, c, HSI_subwindow_size, image_id, HSI_image_dir, image_name, dataset, c, plot_mask=False)
                
                #BGR to RGB
                img = cv.imread(rgb_image_path)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                
                
                rgb_subwindow_size = (256, 256)
                
                rgb_subwindow, rgb_new_annotation = extract_subwindow(img, rgb_new_annotation, c, rgb_subwindow_size, image_id, rgb_image_dir, image_name, dataset, c, plot_mask=False)
                #print(rgb_new_annotation)
                #extract_subwindow(original_img, new_annotation, new_id, window_size, img_id, image_dir, dataset, plot_mask=False):
                #subwindow, new_annotation, mask = extract_subwindow(img, subwindow_size, image_id, image_dir, dataset)
                
                #c = image_id
                
                #subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
                HSI_output_path = os.path.join(HSI_image_dir, "windows\PLS_eval_img") 
                np.save(HSI_output_path + "\\" +  f"{c}_window_{image_name}.npy",HSI_subwindow)
                
                
                rgb_output_path = os.path.join(HSI_image_dir, "windows\PLS_eval_img_rgb") 
                subwindow = cv.cvtColor(rgb_subwindow, cv.COLOR_RGB2BGR)
                cv.imwrite(rgb_output_path + "\\" + f"{c}_window_{image_name}.jpg", subwindow)
                #np.save(rgb_output_path + f"window{c}.jpg", rgb_subwindow)
                c += 1
                print(f"Generating cropped-image: {c} out of {n}")
    export_json(HSI_new_annotation, os.path.join(HSI_image_dir, "windows") + "/COCO_HSI_windowed.json")
    #export_json(rgb_new_annotation,r"C:\Users\jver\Desktop\Test\rgb/COCO_rgb_windowed.json")
   
