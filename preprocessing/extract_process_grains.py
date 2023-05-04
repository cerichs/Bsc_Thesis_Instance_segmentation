from .preprocess_image import binarization, spectral_test

import numpy as np
from skimage.draw import polygon
import os
import re


def find_imgs(dataset, hyp_folder, pseudo_folder):
    
    hyper_path = []
    pseudo_rgbs = []
    img_name = []
    ids = []
    
    def get_window_number(filename):
        return int(re.findall(r'\d+', filename)[0])

    for image_id in range(len(dataset["images"])):
       
        # Extracting file-names in dataset
        image_names = dataset["images"][image_id]["file_name"]
        
        # Extracting paths to pseudo-rgb-images
        pseudo_rgb_path = os.path.join(pseudo_folder, image_names)
        
        # Extracting paths to hyperspectral-images
        hyperspectral_imgs = os.listdir(hyp_folder)
        
        # Filter out any multiplied or subtracted hyperspectral images
        hyperspectral_imgs = [i for i in hyperspectral_imgs if ( ("Multiplied" not in i) and ("subtracted" not in i) )]
        hyperspectral_imgs = np.unique(hyperspectral_imgs)
        #hyper_path = os.path.join(hyp_folder, image_name)

        # As the coco-dataset has initial-index 0, we start by this
        image_id += 1
        
        # Gaining pseudo-rgb-images with hyperspectral-extension to easier find linked files.
        pseudo_name = pseudo_rgb_path.split("\\")[-1].split("/")[-1].split(".")[0] + ".npy"
        
        # Loop over each hyperspectral image and append the path to the matching hyperspectral image
        if pseudo_name in hyperspectral_imgs:
            
            
            [hyper_path.append(os.path.join(hyp_folder, i)) for i in hyperspectral_imgs if pseudo_name in i]
            pseudo_rgbs.append(pseudo_rgb_path)
            img_name.append(pseudo_name.split(".")[0] + ".jpg")
            ids.append(image_id)
        else:
            continue
    hyper_path = sorted(np.unique(hyper_path), key=get_window_number)
    
    return hyper_path, pseudo_rgbs, img_name, ids


def process_data(dataset, hyper_imgs, pseudo_imgs, img_names, image_ids, whole_img=False):
    
    #HSI = [spectral_test(hyper_imgs[image]) for image in range(len(hyper_imgs))]
    HSI = [np.load(hyper_imgs[image]) for image in range(len(hyper_imgs))]
    g_masks = []
    
    for i in range(len(hyper_imgs)):
        if whole_img:
            _, grains_mask = binarization(pseudo_imgs[i])
            g_masks.append([grains_mask])
            split = "whole_img"
        else:
            grains_mask, name, image_id = extract_binary_kernel(dataset, pseudo_imgs[i], img_names[i])
            g_masks.append(grains_mask)  
            split = "grain"
            
    ids = []
    X = []
    X_median = []
    y = []
    
    for images in range(len(g_masks)):
        pixel_avg, label, img_id = pixel_average(HSI[images], g_masks[images], image_ids[images], img_names[images])
        pixel_medis, label, img_id = pixel_median(HSI[images], g_masks[images], image_ids[images], img_names[images])
        ids.append(img_id)
        X.append(pixel_avg)
        X_median.append(pixel_medis)
        y.append(label)
    return ids, X, X_median, y, HSI, split




def extract_binary_kernel(dataset, pseudo_rgbs_path, img_name):
    """
    Extract binary grain masks for an image in the given dataset.
    
    Parameters:
    dataset (dict): A COCO-style dataset dictionary containing image information and annotations.
    pseudo_rgbs_path (str): Path to the corresponding pseudo-RGB image for the given image.
    img_name (str): Name of the image for which to extract the binary grain masks. Used in computing the one-hot-encoding of the labels
    
    Returns:
    grains (list): List of binary grain masks for the image.
    name (list): List of image names for the image.
    ids (list): List of image IDs for the image.
    """
    
    grains = []
    name = []
    ids = []
    
    # Extract image and segmentation information from the dataset dictionary
    anno_img = dataset["images"]
    segment = dataset["annotations"]
    
    # Get the image ID for the given image name
    img_id = [anno_img[j]["id"] for j in range(len(anno_img)) if anno_img[j]["file_name"] == img_name][0]
    
    # Get the segmentation information for the given image ID
    segmentation = [segment[k] for k in range(len(segment)) if segment[k]["image_id"] == img_id]
    
    # Load the pseudo-RGB image and generate a binary mask for it
    im, mask = binarization(pseudo_rgbs_path)
    
    for annotation_numb in range(len(segmentation)):
        print(f"Currently processing grain-number :  {annotation_numb+1} out of {len(segmentation)}")
        
        mini_img = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
        x, y = segmentation[annotation_numb]["segmentation"][0][0::2], segmentation[annotation_numb]["segmentation"][0][1::2]

        
        # Generate a binary mask for the annotation
        for x_x, y_y in zip(x, y):
            
            x_x, y_y = int(x_x), int(y_y)
            mini_img[y_y, x_x] = True
            
        binary_grain_mask = mini_img.astype(int)
        
        # Fill in the polygonal region of the annotation
        row, col = polygon(y, x, mini_img.shape)
        binary_grain_mask[row, col] = 255
        
        # Append the binary grain mask, image name, and image ID to their respective lists
        grains.append(binary_grain_mask)
        name.append(img_name)
        ids.append(img_id)
        
    return grains, name, ids



def pixel_average(hyperspectral_image, binary_mask, image_id, image_name):
    """
    Calculate the average pixel values for each binary grain mask in the given hyperspectral image.
    
    Parameters:
    hyperspectral_image (ndarray): A hyperspectral image as a 3D numpy array.
    binary_mask (list): A list of binary masks for each grain in the image.
    image_id (int): The ID of the image.
    image_name (str): The name of the image.
    
    Returns:
    pixel_averages (list): List of average pixel values for each grain mask in the image.
    one_hot (list): List of one-hot-encoded labels for each grain mask in the image.
    image_id (int): The ID of the image.
    """
    
    pixel_averages = []
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4", "Wheat_H5", "Wheat_Halland", "Wheat_Oland", "Wheat_Spelt"]
    one_hot = []
    id_img = []
    
    # Loop through each binary mask
    for mask in binary_mask:
        # Extract the pixel values for each grain mask
        grains = hyperspectral_image[mask == 255, :] # Only consider pixels with value 255
        
        # Calculate the mean spectra-values for each grain mask
        grain_pixel_average = np.mean(grains, axis=0)
        pixel_averages.append(grain_pixel_average)
        
        # Generate one-hot-encoded labels for each grain mask
        labels = [int(names in image_name) for names in class_list]
        one_hot.append(labels)
        
        # Gain image_id of each mask
        id_img.append(image_id)
   
    return pixel_averages, one_hot, id_img


def pixel_median(hyperspectral_image, binary_mask, image_id, image_name):
    """
    Calculate the average pixel values for each binary grain mask in the given hyperspectral image.
    
    Parameters:
    hyperspectral_image (ndarray): A hyperspectral image as a 3D numpy array.
    binary_mask (list): A list of binary masks for each grain in the image.
    image_id (int): The ID of the image.
    image_name (str): The name of the image.
    
    Returns:
    pixel_averages (list): List of average pixel values for each grain mask in the image.
    one_hot (list): List of one-hot-encoded labels for each grain mask in the image.
    image_id (int): The ID of the image.
    """
    
    pixel_medians = []
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4", "Wheat_H5", "Wheat_Halland", "Wheat_Oland", "Wheat_Spelt"]
    one_hot = []
    id_img = []
    
    # Loop through each binary mask
    for mask in binary_mask:
        # Extract the pixel values for each grain mask
        grains = hyperspectral_image[mask == 255, :] # Only consider pixels with value 255
        
        # Calculate the mean spectra-values for each grain mask
        grain_pixel_median = np.median(grains, axis=0)
        pixel_medians.append(grain_pixel_median)
        
        # Generate one-hot-encoded labels for each grain mask
        labels = [int(names in image_name) for names in class_list]
        one_hot.append(labels)
        
        # Gain image_id of each mask
        id_img.append(image_id)
   
    return pixel_medians, one_hot, id_img




def extract_specifics(dataframe, grain_class, row_indices=None):
    """
    Extracts data from a pandas DataFrame based on a given grain class label and optional row indices.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing the spectral data and labels.
        grain_class (str): Grain class label for which to extract data.
        row_indices (list): Optional list of start and end indices for rows to extract.

    Returns:
        pandas.DataFrame: DataFrame containing the extracted data.

    Raises:
        ValueError: If the grain_class argument is not valid.

    """
    # Define the class labels and corresponding indices
    class_labels = {
        "rye": "[1, 0, 0, 0, 0, 0, 0, 0]",
        "h1": "[0, 1, 0, 0, 0, 0, 0, 0]",
        "h3": "[0, 0, 1, 0, 0, 0, 0, 0]",
        "h4": "[0, 0, 0, 1, 0, 0, 0, 0]",
        "h5": "[0, 0, 0, 0, 1, 0, 0, 0]",
        "halland": "[0, 0, 0, 0, 0, 1, 0, 0]",
        "oland": "[0, 0, 0, 0, 0, 0, 1, 0]",
        "spelt": "[0, 0, 0, 0, 0, 0, 0, 1]"
    }

    # Check if the given grain_class is valid, if not raise an error
    if grain_class.lower() not in class_labels:
        raise ValueError("Invalid grain class label. Valid options are: 'rye', 'h1', 'h3', 'h4', 'h5', 'halland', 'oland', 'spelt'.")

    # Extract the rows that match the given grain_class label
    result = dataframe.loc[dataframe['label'] == class_labels[grain_class.lower()]].reset_index(drop=True)

    # If row_indices are given, extract only those rows
    if row_indices:
        try:
            result = result.iloc[row_indices[0]:row_indices[1], ]
        except:
            raise ValueError("Invalid row indices. Should be a list of start and end indices.")

    return result





