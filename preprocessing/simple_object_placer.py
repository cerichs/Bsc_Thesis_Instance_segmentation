import numpy as np
import cv2 as cv
import os
import sys
from tqdm import tqdm

sys.path.append("..")

from .Display_mask import load_coco, load_annotation, find_image, draw_img
from .crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
from .preprocess_image import spectral_test, binarization
from two_stage.watershed_2_coco import empty_dict, export_json



def find_x_y(larger_image,smaller_image,annotation,height,width):
    """
    Find the x and y coordinates to place a smaller_image in a larger_image.
    
    Args:
        larger_image: The larger image where the smaller_image will be placed.
        smaller_image: The smaller image to be placed in the larger_image.
        annotation: The annotation data for the smaller_image.
        height: The height of the smaller_image.
        width: The width of the smaller_image.

    Returns:
        tuple: x, y coordinates and a boolean indicating if the placement is successful.
    """
    count = 0
    while True:
        x = np.random.randint(0, (larger_image.shape[1] - smaller_image.shape[1]))
        y = np.random.randint(0, (larger_image.shape[0] - smaller_image.shape[0]))
        x, y = edge_grain(annotation, height, width, x, y, larger_image)
        temp = larger_image[y:y + smaller_image.shape[0], x:x + smaller_image.shape[1]]
        overlap_amount = np.sum(np.multiply(temp, smaller_image))

        if overlap_amount < 25000:  # TODO: make it a ratio of kernel, as smaller kernels constantly overlapped
            return x, y, True
        count += 1
        if count > 50:
            return x, y, False


def edge_grain(annotation,height,width,x,y,larger_image):
    """
    Prevents the smaller image from being placed too close to the edges of the larger image.
    
    Args:
        annotation: The annotation data for the smaller_image.
        height: The height of the smaller_image.
        width: The width of the smaller_image.
        x: The x-coordinate of the smaller_image placement.
        y: The y-coordinate of the smaller_image placement.
        larger_image: The larger image where the smaller_image will be placed.
        
    Returns:
        tuple: The adjusted x and y coordinates for the placement.
    """
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
    """
    Return the next available annotation id in the COCO dataset.
    
    Args:
        coco_dict: The COCO dataset dictionary.

    Returns:
        The next available image id.
    """
    return len(coco_dict['annotations'])+1

def coco_next_img_id(coco_dict):
    """
    Return the next available image id in the COCO dataset.
    
    Args:
        dataset: The COCO dataset
        smaller_image: The smaller image to be placed in the larger_image.
        image_id: The id of the image in the dataset.
        annotation_numb: The number of the annotation for the image.
        x: The x-coordinate of the placement.
        y: The y-coordinate of the placement.

    Returns:
        The new annotation coordinates.
    """
    return len(coco_dict['images'])+1

def coco_new_anno_coords(dataset,image_id,annotation_numb,x,y):
    """
    Calculate the new annotation coordinates for the COCO dataset.
    
    Args:
        larger_image: The larger image where the smaller_image will be placed
        smaller_image: The smaller image to be placed in the larger_image
        annotation: The annotation data for the smaller_image
        height: The height of the smaller_image
        width: The width of the smaller_image

    Returns:
        tuple: x, y coordinates and a boolean indicating if the placement is successful.
    """
    width = min(dataset['annotations'][annotation_numb]['segmentation'][0][0::2])
    height = min(dataset['annotations'][annotation_numb]['segmentation'][0][1::2])
    new_annote = [None]*len(dataset['annotations'][annotation_numb]['segmentation'][0]) # new list the same size
    new_annote[0::2] = [z - width for z in dataset['annotations'][annotation_numb]['segmentation'][0][0::2]] # rescale to start at 0
    new_annote[1::2] = [z - height for z in dataset['annotations'][annotation_numb]['segmentation'][0][1::2]] # rescale to start at 0
    new_annote[0::2] = [z + x for z in new_annote[0::2]] # rescale for new img
    new_annote[1::2] = [z + y for z in new_annote[1::2]] # rescale for new img
    return new_annote

def coco_new_bbox(x,y,dataset,image_id,annotation_numb):
    """
    Calculate the new bounding box for the COCO dataset.
    
    Args:
        larger_image: The larger image where the smaller_image will be placed
        smaller_image: The smaller image to be placed in the larger_image
        annotation: The annotation data for the smaller_image
        height: The height of the smaller_image
        width: The width of the smaller_image

    Returns:
        tuple: x, y coordinates and a boolean indicating if the placement is successful.
    """
    annote = coco_new_anno_coords(dataset,image_id,annotation_numb,x,y)
    width = max(annote[0::2])-min(annote[0::2])
    height = max(annote[1::2])-min(annote[1::2])
    return [x,y,width,height]

def create_synt_images(annotation_path, HSI_image_dir, image_dir, n=100):
    # Load the COCO dataset
    dataset = load_coco(annotation_path)
    
    # Create an empty COCO-like dictionary
    dict_coco = empty_dict()
    
    # Copy the categories, info, and licenses from the original dataset
    dict_coco['categories'] = dataset['categories']
    dict_coco['info'] = dataset['info']
    dict_coco['licenses'] = dataset['licenses']
    
    # Define the list of class IDs
    class_list = [1412692, 1412693, 1412694, 1412695, 1412696, 1412697, 1412698, 1412699, 1412700]
    #              [Rye_midsummer, Wheat_H1, Wheat_H3, Wheat_H4, Wheat_H5, Wheat_Halland, Wheat_Oland, Wheat_Spelt, Foreign]
    
    for c in tqdm(range(n)):
        # Create a background image
        background = np.zeros((256, 256, 3), dtype=np.uint8)
        background_hsi = np.zeros((256, 256, 102), dtype=np.float64)
        background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
        
        max_tries = 200  # Amount of kernels to randomly sample and try to place
        class_check = [0] * 8
        j = 0
        while j < max_tries:
            # Randomly select an annotation
            annotation_numb = np.random.randint(0, len(dataset['annotations']))
            
            # Find the image name and ID for the selected annotation
            image_name, image_id = find_image(dataset, annotation_numb)
            
            # Load the annotation and bbox for the selected annotation
            bbox, annotation = load_annotation(dataset, annotation_numb, image_id)
            
            # Fill the mask to get the cropped image
            cropped_im = fill_mask(dataset, image_id, annotation, image_name, image_dir)
            height, width = cropped_im.shape[0], cropped_im.shape[1]
            
            # Crop the image based on the mask
            cropped = crop_from_mask(dataset, annotation_numb, cropped_im)
            
            # Find the position to place the cropped image on the background
            x, y, keep = find_x_y(background, cropped, annotation, height, width)
            
            # Specify the maximum number of each kernel type to be generated
            max_numb = 8
            if keep and (dataset['annotations'][annotation_numb]['category_id'] != 1412700) and (class_check[class_list.index(dataset['annotations'][annotation_numb]['category_id'])] < max_numb):
                # Overlay the cropped image on the background
                background = overlay_on_larger_image(background, cropped, x, y)
                
                # Load the HSI image
                HSI_image_path = os.path.join(HSI_image_dir, image_name)
                if not HSI_image_path.endswith(".npy"):
                    HSI_image_path = HSI_image_path[:-3]+"npy"
               
                # Load the HSI image and resize it
                HSI_img = spectral_test(HSI_image_path)
                resized_hsi = cv.resize(HSI_img, (cropped.shape[1], cropped.shape[0]))
                
                # Extract the grains from the resized HSI image using the cropped mask
                grains = cv.bitwise_and(resized_hsi, resized_hsi, mask=np.uint8(cropped[:, :, 0]))
                
                # Create a temporary array to hold the updated HSI values
                temp = np.zeros((grains.shape[0], grains.shape[1], 102), dtype=np.float64)
                temp = temp.astype(grains.dtype)
                temp = background_hsi[y:y+grains.shape[0], x:x+grains.shape[1]]  # Selecting a window of the image to edit
                temp[grains > 0] = 0  # All the places where the object is, is set to 0. Where the mask is 0, does remains unchanged from the larger_image
                temp = np.add(temp, grains * (grains > 0), out=temp, casting="unsafe")
                background_hsi = background_hsi.astype(grains.dtype)
                background_hsi[y:y+grains.shape[0], x:x+grains.shape[1]] = temp
                
                # Append the annotation information to the COCO-like dictionary
                dict_coco['annotations'].append({
                    'id': coco_next_anno_id(dict_coco),
                    'image_id': coco_next_img_id(dict_coco),
                    'segmentation': [coco_new_anno_coords(dataset, image_id, annotation_numb, x, y)],
                    'iscrowd': 0,
                    'bbox': coco_new_bbox(x, y, dataset, image_id, annotation_numb),
                    'area': dataset['annotations'][annotation_numb]['area'],
                    'category_id': dataset['annotations'][annotation_numb]['category_id']
                })
                
                # Increment the class count
                class_check[class_list.index(dataset['annotations'][annotation_numb]['category_id'])] += 1
            else:
                pass
            j += 1
        
        # Save the background image
        background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
        output_folder_rgb = os.path.join(HSI_image_dir, "windows", "PLS_eval_img_rgb")
        print(output_folder_rgb)
        print("")
        print(os.path.join(output_folder_rgb, f"Synthetic_{c}.npy"))
        cv.imwrite(os.path.join(output_folder_rgb, f"Synthetic_{c}.jpg"), background)
        
        # Display the background image
        import matplotlib.pyplot as plt
        plt.imshow(background)
        plt.show()
        
        # Save the background HSI image
        output_folder_HSI = os.path.join(HSI_image_dir, "windows", "PLS_eval_img")
        np.save(os.path.join(output_folder_HSI, f"Synthetic_{c}.npy"), background_hsi)
        
        # Display the background HSI image
        plt.imshow(background_hsi[:, :, 0])
        plt.show()
        
        # Append the image information to the COCO-like dictionary
        dict_coco['images'].append({
            'id': c + 1,
            'file_name': f"Synthetic_{c}.jpg",
            'license': 1,
            'height': background.shape[0],
            'width': background.shape[1]
        })
    
    # Export the COCO-like dictionary as a JSON file
    export_json(dict_coco, os.path.join(HSI_image_dir, "windows" "COCO_synthetic.json"))
    
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
