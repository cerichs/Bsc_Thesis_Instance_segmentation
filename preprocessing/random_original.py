

import os
os.chdir(r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing")

from Display_mask import load_coco, load_annotation, find_image, draw_img
from crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
from watershed_2_coco import empty_dict, export_json
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.draw import polygon




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
        return x,y #ændre i masken hvis kantkorn

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

def coco_new_bbox(x,y,dataset,image_id,annotation_numb): #giv maske
    annote = coco_new_anno_coords(dataset,image_id,annotation_numb,x,y)
    width = max(annote[0::2])-min(annote[0::2])
    height = max(annote[1::2])-min(annote[1::2])
    return [x,y,width,height]








def generate_subwindow_image(dataset, image_dir, size=(128, 128, 3)):
    annotation_numb = np.random.randint(0, len(dataset['annotations']))
    image_name, image_id = find_image(dataset, annotation_numb)
    image_path = os.path.join(image_dir, image_name)
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    subwindow, x, y = get_subwindow(image, size=(size[0], size[1]))
    masks = []
    for ann in dataset['annotations']:
        if ann['image_id'] == image_id:
            rr, cc = polygon(ann['segmentation'][0][1::2], ann['segmentation'][0][0::2])
            mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
            mask[rr, cc] = 1
            mask = mask[y:y + size[1], x:x + size[0]]
            masks.append(mask)
    return subwindow, masks






def create_subwindow_image(original_image, window_size, annotations=None):
    if annotations is None:
        annotations = []

    subwindow = np.zeros(window_size, dtype=np.uint8)
    subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)

    grain_mask = create_grain_mask(annotations, original_image.shape)
    cropped_image = crop_grain_from_mask(original_image, grain_mask)
    x, y = find_non_overlapping_position(subwindow, cropped_image)

    if x is not None and y is not None:
        subwindow = overlay_on_larger_image(subwindow, cropped_image, x, y)

    return subwindow

def create_grain_mask(annotation, image_shape):
    rr, cc = polygon(annotation[1::2], annotation[0::2])
    mask = np.zeros(image_shape[:2], dtype=bool)
    mask[rr, cc] = True
    return mask

def crop_grain_from_mask(image, mask):
    y_min, y_max, x_min, x_max = find_bounding_box(mask)
    cropped_image = image[y_min:y_max, x_min:x_max].copy()
    cropped_image[~mask[y_min:y_max, x_min:x_max]] = 0
    return cropped_image





def extract_subwindow(original_img, window_size, img_id, image_dir, dataset):
    window_height, window_width = window_size

    top_left_x = np.random.randint(0, original_img.shape[1] - window_width)
    top_left_y = np.random.randint(0, original_img.shape[0] - window_height)

    bottom_right_x = top_left_x + window_width
    bottom_right_y = top_left_y + window_height

    subwindow = original_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    new_annotation = empty_dict()
    
    
    for ann in dataset['annotations']:
        if ann['image_id'] == image_id:
            new_coords = []
            for coord_x, coord_y in zip(ann['segmentation'][0][0::2], ann['segmentation'][0][1::2]):
                new_x = coord_x - top_left_x
                new_y = coord_y - top_left_y
                
                new_coords.extend([new_x, new_y])
                
                """
                if 0 <= new_x < window_width and 0 <= new_y < window_height:
                    new_coords.extend([new_x, new_y])
                """
                #if new_x < 0:
                    
            if new_coords[0::2] < 0 & new_coords[1::2] < 0:
                x_root = np.array(new_coords[0::2])
                y_root = np.array(new_coords[1::2])
                
                
                x_root = np.where(x_root==0)[0]
                y_root = np.where(y_root==0)[0]
                
                if x_root < window_width and y_root < window_height:
                    
                
            
            if new_coords[1::2] < 0:
                y_root = new_coords[1::2].index(0)
                y_root = np.array(y_root)
                
                
            ann["bbox"][0] = ann["bbox"][0] - top_left_x
            ann["bbox"][1] = ann["bbox"][1] - top_left_y 



            if len(new_coords) > 0:
                new_annotation["annotations"].append({'id': ann['id'],
                                       'image_id': ann['image_id'],
                                       'segmentation': [new_coords],
                                       'iscrowd': ann['iscrowd'],
                                       'bbox': ann['bbox'],
                                       'area': ann['area'],
                                       'category_id': ann['category_id']})





    new_annotation['images'].append({'id':img_id,
                            'file_name': f"window{img_id}.jpg",
                            'license':1,
                            'height':subwindow.shape[0],
                            'width':subwindow.shape[1]})
    
    

    return subwindow, new_annotation


################## MAIN ##################

#annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
annotation_path = r"C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Training\COCO_Training.json"
#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
image_dir = r'C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Training\images/'

dataset = load_coco(annotation_path)
annotation_numb = 1   ### choose specific image

# Get image-info from JSON
image_name, image_id = find_image(dataset, annotation_numb)
bbox, annotation = load_annotation(dataset, annotation_numb, image_id)
image_path = os.path.join(image_dir, image_name)


#BGR to RGB
img = cv.imread(image_path)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


subwindow_size = (128, 128)

subwindow, new_annotation = extract_subwindow(img, subwindow_size, image_id, image_dir, dataset)
c = annotation_numb

export_json(new_annotation)

subwindow = cv.cvtColor(subwindow, cv.COLOR_BGR2RGB)
cv.imwrite(f"window{c}.jpg",subwindow)


image_dir = "C:/Users/admin/Desktop/bachelor/Bsc_Thesis_Instance_segmentation/preprocessing/"
#cv.destroyWindow(f"/window{c}.jpg")

annote_ids = []
for i in range(len(new_annotation['annotations'])):
    if new_annotation['annotations'][i]['image_id']==image_id:
        annote_ids.append(i)
draw_img(new_annotation,image_id,annote_ids, image_dir)
# Generate the sub-window image
#subwindow_image = create_subwindow_image(img, annotations=annotation)




# Generate the sub-window image
#subwindow_image = create_subwindow_image(large_image, window_size=subwindow_size annotations=annotations)


#, masks = generate_subwindow_image(dataset, image_dir, size=subwindow_size)





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
