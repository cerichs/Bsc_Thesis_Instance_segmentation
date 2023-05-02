import numpy as np
import cv2 as cv
import sys

sys.path.append("..")

from .Display_mask import load_coco, load_annotation, find_image, draw_img
from .crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
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

if __name__=="__main__":
    #annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
    annotation_path = 'C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/COCO_Test.json'
    #image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    image_dir = 'C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/images/'
    dataset = load_coco(annotation_path)
    dict_coco = empty_dict()
    dict_coco['categories']=dataset['categories']
    dict_coco['info']=dataset['info']
    dict_coco['licenses']=dataset['licenses']
    class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
    #           [Rye_midsummer, Wheat_H1, Wheat_H3,  Wheat_H4,   Wheat_H5, Wheat_Halland,  Wheat_Oland, Wheat_Spelt, Foreign]
                
    for c in range(200):
        background = np.zeros((256,256,3),dtype = np.uint8)
        background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
        max_tries=200 # amount of kernel to randomly sample and try to place
        class_check= [0]*8
        j=0
        while(j<max_tries):
            annotation_numb = np.random.randint(0,len(dataset['annotations']))
            
            image_name, image_id = find_image(dataset, annotation_numb)
            bbox, annotation = load_annotation(dataset, annotation_numb, image_id)
            cropped_im = fill_mask(dataset,image_id, annotation, image_name,image_dir)
            height, width = cropped_im.shape[0], cropped_im.shape[1]
            cropped = crop_from_mask(dataset, annotation_numb, cropped_im)
            x, y, keep = find_x_y(background, cropped,annotation, height,width)
            
            #Specifying the max of each kernel type to be generated:
            max_numb = 8
            if keep and (dataset['annotations'][annotation_numb]['category_id']!=1412700) and (class_check[class_list.index(dataset['annotations'][annotation_numb]['category_id'])] < max_numb):
                background = overlay_on_larger_image(background,cropped,x,y)
                dict_coco['annotations'].append({'id':coco_next_anno_id(dict_coco),
                                      'image_id':coco_next_img_id(dict_coco),
                                      'segmentation': [coco_new_anno_coords(dataset,image_id,annotation_numb,x,y)],
                                      'iscrowd':0,
                                      'bbox': coco_new_bbox(x,y,dataset,image_id,annotation_numb), #mangler x,y
                                      'area':dataset['annotations'][annotation_numb]['area'],
                                      'category_id':dataset['annotations'][annotation_numb]['category_id']
                                      })
                class_check[class_list.index(dataset['annotations'][annotation_numb]['category_id'])] += 1
            else:
                pass
            j+=1
        print(class_check)
        background= cv.cvtColor(background, cv.COLOR_BGR2RGB)
        cv.imwrite(f"images/Synthetic_{c}.jpg",background)
        dict_coco['images'].append({'id':c+1,
                                    'file_name': f"Synthetic_{c}.jpg",
                                    'license':1,
                                    'height':background.shape[0],
                                    'width':background.shape[1]})
    export_json(dict_coco,"COCO_balanced_1k_val.json")
    
    plot_mask = False
    if plot_mask:
        ### Extracting name of the particular grain-type and counting instances in image
        for new_id in range(1, 11):
            ground_truth = 0
            name = []
            for annotations in dict_coco["annotations"]:
                if annotations["image_id"]==new_id:
                    ground_truth += 1
                    for categories in dict_coco["categories"]:
                        if categories["id"] == annotations["category_id"]:
                            #print(annotations["category_id"])
                            name.append(categories["name"])
            #name = set(name)
            unique, counts = np.unique(name, return_counts=True)
            print("")
            print(f"The following grain-type being analysed is:  {dict(zip(unique, counts))}   with image_id:  {new_id}")
            print("")
            print(f"The ground-truth amount of kernels in the image is:  {ground_truth}")
                
        #c = 0 

        for new_id in range(1, 11):
            annote_ids = []
            
            #print (new_annotation['annotations'] )
            for i in range(len(dict_coco['annotations'])):
                #print(i)
                if dict_coco['annotations'][i]['image_id']==new_id:
                    #print(image_id)
                    annote_ids.append(i)
                else:
                    continue
            
            image_dir2 = r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\images/"
            draw_img(dict_coco, new_id, annote_ids, image_dir2)
            
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
