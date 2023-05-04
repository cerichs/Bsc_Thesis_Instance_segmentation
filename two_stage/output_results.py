import sys
sys.path.append("..")
from preprocessing.simple_object_placer import coco_next_anno_id

import pandas as pd
import numpy as np
from skimage.draw import polygon





def create_dataframe(mask_ids, labels, pixel_averages, split):
    """
    Create a pandas dataframe containing the average pixel values for each grain mask in the given images,
    along with the corresponding maskIDs and one-hot-encoded labels.
    
    Parameters:
    maks_ids (list): A list of mask_IDs.
    labels (list): A list of one-hot-encoded labels for each grain mask in the images.
    pixel_averages (list): A list of average pixel values for each grain mask in the images.
    split (str): A string indicating whether the data is for training, validation, or testing.
    
    Returns:
    df (DataFrame): A pandas dataframe containing the average pixel values for each grain mask in the images,
    along with the corresponding image IDs and one-hot-encoded labels.
    """
    pixel_avg = pixel_averages.copy()
    if isinstance(pixel_avg, list):
        # Creating dataframe with 103 columns - 1 label and 102 for the channels
        df = pd.DataFrame(columns=[None]*104)
        df.columns = ["image_id"] + ["label"] + [f"wl{i}" for i in range(1, 103)]
        
        for image in range(len(mask_ids)):
            for mask in range(len(mask_ids[image])):
                # Creating list of length 104 with mask_ids, one-hot label, and respective 102 pixel-averages
                temp = [mask_ids[image][mask]] + [labels[image][mask]] + pixel_avg[image][mask].tolist()
            
                # Appending this to the dataframe
                df.loc[len(df)] = temp
                
    elif isinstance(pixel_avg, pd.DataFrame):
        flattened_ids = [val for sublist in mask_ids for val in sublist]
        flattened_labels = [val for sublist in labels for val in sublist]
        pixel_avg.insert(0, 'labels', flattened_labels)
        pixel_avg.insert(0, 'image_id', flattened_ids)
        df = pixel_avg
        df.columns = ["image_id"] + ["label"] + [f"wl{i}" for i in range(1, 103)]
            
    # Saving dataframe for later use
    df.to_csv(f"two_stage/pls_results/Pixel_grain_avg_dataframe_{split}.csv", index=False)
        
    return df


def add_2_coco(dict_coco, dataset, annotations, pseudo_img, class_id):
    
    for image in dataset["images"]:
        
        if image["file_name"] in pseudo_img:
            ids = image["id"]
            
            dimensions = (image["height"],image["width"])
            
    class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
    start_x = min(annotations[0::2])
    start_y = min(annotations[1::2])
    width = max(annotations[0::2])-start_x
    height = max(annotations[1::2])-start_y
    mini_img = np.zeros(dimensions,dtype=bool)
    x, y = (annotations[0::2]),(annotations[1::2])
    for x_x,y_y in zip(x,y):
        x_x, y_y = int(x_x), int(y_y)
        mini_img[y_y,x_x]=True
    img=mini_img.astype(int)

    row, col = polygon(y, x, img.shape)
    img[row,col] = 1
    
    dict_coco['annotations'].append({'id': coco_next_anno_id(dict_coco),
                              'image_id': ids,
                              'segmentation': [annotations],
                              'iscrowd': 0,
                              'bbox': [start_x, start_y, width, height],
                              'area':int(np.sum(img)),
                              'category_id':class_list[class_id]
                              })
    return dict_coco