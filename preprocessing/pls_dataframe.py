
#### PLS 
from Display_mask import load_coco, load_annotation, find_image, draw_img
from crop_from_mask import crop_from_mask, fill_mask,overlay_on_larger_image
from watershed_2_coco import empty_dict, export_json
from simple_object_placer import coco_next_anno_id

import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from watershed_v2 import preprocess_image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import cv2 as cv
import skimage
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt


#centrum = (bbox[0][2] / 2, bbox[0][3]/2)
# en pixel rundt om centrum 
# pixel_c_box = [centrum[0]-1, centrum[1]-1, 1, 1]



"""

image_name = dataset["images"][image_id]["file_name"]
image_id = image_id + 1

# Get image-info from JSON
image_path = os.path.join(image_dir, image_name)


#BGR to RGB
img = cv.imread(image_path)


def pixel_average(hyperspectral_image, mask):
    # Calculate the pixel average for each unique label in the mask
    unique_labels = np.unique(mask)
    pixel_averages = []

    for label in unique_labels:
        if label == 0:  # Assuming 0 is the background label
            continue

        grain_mask = (mask == label)
        grain_pixels = hyperspectral_image[grain_mask, :]
        grain_pixel_average = np.mean(grain_pixels, axis=0)
        pixel_averages.append(grain_pixel_average)

    pixel_averages = np.array(pixel_averages)
    return pixel_averages


def extract_image_features_and_labels(images, labels):
    
    unique_image_ids = list(set(images))

    # Preprocess images
    preprocessed_images = [preprocess_image(img) for img in images]

    # Flatten images to convert them into 1D arrays and create a feature matrix 
    X = [img.flatten() for img in preprocessed_images]
    # target class labels are one-hot encoded and form the target matrix Y

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(labels))
    #print(integer_encoded)

    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    #print(Y)
    
    return X, Y
    

def train_pls_models(images, image_ids, n_components=10):
    
    pls_models = {}
    for image_id in image_ids:
        image = images[image_id]
        X_image, Y_image = extract_image_features_and_labels(image)
        pls_model = PLSRegression(n_components=n_components)
        pls_model.fit(X_image, Y_image)
        pls_models[image_id] = pls_model
    return pls_model




def evaluate_pls_models(pls_models, images, test_image_ids):
    mse_scores = []
    r2_scores = []

    for image_id in test_image_ids:
        pls_model = pls_models[image_id]
        image = images[image_id]

        X_test_image, Y_test_image = extract_image_features_and_labels(image)
        Y_pred_image = pls_model.predict(X_test_image)

        mse = mean_squared_error(Y_test_image, Y_pred_image)
        r2 = r2_score(Y_test_image, Y_pred_image)

        mse_scores.append(mse)
        r2_scores.append(r2)

    return mse_scores, r2_scores

"""








def spectral_test(img_path):
    #PATH = r"C:\Users\admin\Downloads\Sparse_Series1_20_09_08_07_47_28.npy"
    np.seterr(divide = 'ignore') 
    #np.seterr(divide = 'warn') 
    
    array = np.load(img_path)
    #test = np.where(array>0)
    
    
    hyp_orig = img_path.split("\\")[-1].split(".")[0]
    
    sub = "subtracted_" + hyp_orig + ".npy"
    mult = "Multiplied_" + hyp_orig + ".npy"
    
    sub = np.load(os.path.join(os.path.split(img_path)[0], sub))
    mult = np.load(os.path.join(os.path.split(img_path)[0], mult))
    
    
    array = array / mult + sub
    
    
    temp = -np.log10(array)
    img_discarded = temp[:,:,9:213]
    
    new_img=np.zeros((img_discarded.shape[0],img_discarded.shape[1],int(img_discarded.shape[2]/2)))
    for i in range(1,(int(len(img_discarded[0,0,:])/2)+1),1):
        #print(i*2-2)
        #print(i*2-1)
        new_img[:,:,i-1]=(img_discarded[:,:,i*2-2]+img_discarded[:,:,(i*2-1)])/2
    #imageio.imwrite("C:/Users/Cornelius/Downloads/e4Wr5LFI4L/Training/Wheat_H1" + str(0)  + '.tiff', new_img)
    
    #pseudo_rgb = np.zeros((temp.shape[0],temp.shape[1],3))
    """
    new_img_2 = new_img.copy()
    print("test")
    for i in range(new_img.shape[2]):
        #new_img[:,:,i] = (new_img[:,:,i] - np.mean(new_img[:,:,i])) / np.std(new_img[:,:,i])
        #new_img[:,:,i] = (new_img[:,:,i]-new_img[:,:,i].min())/(new_img[:,:,i].max()-new_img[:,:,i].min())
        new_img_2[:,:,i]=((255-0)/(new_img_2[:,:,i].max()-new_img_2[:,:,i].min()))*(new_img_2[:,:,i]-new_img_2[:,:,i].min())
    """
    return new_img




def mask(image_dir, image_name):
    grayscale = cv.cvtColor(pseudo_rgb_mask,cv.COLOR_RGB2GRAY) # Convert to Grayscale


    #Converting image to float - depicting gray-scale intensities
    image_g = skimage.util.img_as_float(grayscale)
    # image histogram - used in explaining Otsu's method
    histogram, bin_edges = np.histogram(image_g, bins=256)

    #Otsu's method for thresholding
    auto_tresh = threshold_otsu(image_g) # Determine Otsu threshold
    #Plotting histogram, threshold value and 

    segm_otsu = (image_g > auto_tresh) # Apply threshold
    img_t = segm_otsu.astype(int)*255 # Convert Bool type to 0:255 int
    
    return img_t



def pixel_average(hyperspectral_image, mask):
    # Calculate the pixel average for each unique label in the mask
    unique_labels = np.unique(mask)
    pixel_averages = []

    for label in unique_labels:
        if label == 0:  # Assuming 0 is the background label
            continue
        
        else:
            
            grain_mask = (mask == label)
            grain_pixels = hyperspectral_image[grain_mask, :]
            print(np.amax(grain_pixels))
            grain_pixel_average = np.mean(grain_pixels, axis=0)
            pixel_averages.append(grain_pixel_average)

    #pixel_averages = np.array(pixel_averages)
    
    return pixel_averages



    
    
    
"""
def load_data(coco_path, image_dir, image_id):
    
    dataset = load_coco(path)
    
    
    image_name = dataset["images"][image_id]["file_name"]
    # Get image-info from JSON
    image_path = os.path.join(image_dir, image_name)
    
    label = []
    image_id += 1
    for annotations in dataset["annotations"]:
        if annotations["image_id"]==image_id:
            for categories in dataset["categories"]:
                if categories["id"] == annotations["category_id"]:
                    #print(annotations["category_id"])
                    label.append(categories["name"])
    
    
    #Loading annotation
    annotation = []
    annote_ids = []
    for i in range(len(dataset['annotations'])):
        if dataset['annotations'][i]['image_id']==image_id:
            annote_ids.append(i)
            annotation.append(dataset['annotations'][i]['segmentation'])
            
    for idx in annote_ids:
        bbox, annotation = load_annotation(dataset, idx,image_numb)
        cropped_im = fill_mask(dataset, image_numb, annotation, image_name,image_path)
        cropped = crop_from_mask(dataset,idx,cropped_im)
        pseudo_rgb_mask = overlay_on_larger_image(background,cropped)
    
    return dataset, image_name, labels, annotation, annote_ids
"""

    

if __name__ == "__main__":
    
    #Training_data
    #annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
    train_annotation_path = r"C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Training\COCO_Training.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    train_image_dir = r"C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Training\images/"
    
    #Test_data
    test_annotation_path = r"C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Test\COCO_Test.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    test_image_dir = r"C:\Users\admin\Downloads\DreierHSI_Mar_07_2023_13_24_Ole-Christian Galbo\Test\images/"
    
    
    #Loading data-sets
    #dataset_train, image_names_train, labels_train, annotation, annote_ids = load_data(train_annotation_path, train_image_dir, 0)
    
    #dataset_test = load_coco(test_annotation_path)
    
    #Loading path to hyperspectral image
    hyperspectral_path_train = r"C:\Users\admin\Downloads\hyper\Training"
    
    dataset = load_coco(train_annotation_path)
    
    df = pd.DataFrame( columns = [None]*103)
    df.columns = [f"wl{i}" for i in range(0, 103)]
    
    #range(len(dataset_train["images"]))
    for image_id in range(len(dataset["images"])):
        #Loading path to pseudo_rgb image
        image_name = dataset["images"][image_id]["file_name"]
        pseudo_rgb = os.path.join(train_image_dir, image_name)
        #print(pseudo_rgb)
        
        
        #Retrieving shape of pseudo_rgb - shall be used in creation of background
        pseudo_shape = cv.imread(pseudo_rgb).shape
        background = np.zeros(pseudo_shape,dtype = np.uint8)
        background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
        
        image_id += 1
        #Checking if the same grain-type
        
        hyper_folder = os.path.join(hyperspectral_path_train, "_".join(pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0].split("_")[1:3]))
        
        
        hyperspectral_imgs = os.listdir(hyper_folder)
        hyperspectral_imgs = [i for i in hyperspectral_imgs if (("Multiplied" not in i) and ("subtracted" not in i))]
        
        
        #Check if hyperspectral_id in pseudo_rgb_id
        
        
        #pseudo_rgb.split("\\")[-1].split(".")[0][9+len(hyper_folder.split("\\")[-1])+1:-4]
        print(pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][9+len(hyper_folder.split("\\")[-1])+1:])
        print(f"hyperspectral_imgs  :  {hyperspectral_imgs}")
        #print(hyperspectral_imgs)
        
        if pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][9+len(hyper_folder.split("\\")[-1])+1:] + ".npy" in hyperspectral_imgs:
            hyperspectral_img = os.path.join(hyper_folder, pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][9+len(hyper_folder.split("\\")[-1])+1:] + ".npy")
            
            
            
            #Now we have the full path to hyperspectral image
            #Load spectral image
            spectral_img = spectral_test(hyperspectral_img)
            #Gain mask from pseudo_rgb
            _, mask_h = preprocess_image(pseudo_rgb)
            plt.imshow(cv.imread(pseudo_rgb))
            plt.show()
            plt.imshow(mask_h)
            plt.show()
            
            #pixel_average of spectral_img based on the binary mask
            pixel_avg = pixel_average(spectral_img, mask_h)
            
            #df.loc[len(df)] = [, pixel_avg[0].tolist()]
            
            
            # One-Hot-Encoded Labels
            class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
            
            labels = [int(names in image_name) for names in class_list]
            
            
            temp = [labels] + pixel_avg[0].tolist()
            
            df.loc[len(df)] = temp
            
            
    
            
            #df.append(pixel_avg[0].tolist())
        else:
            continue
                
                
                
                
                
    
 
    
    """
    
       
    fill_mask(dataset,image_id,annotation,image_name,image_path):
    
    
    pixel_averages = pixel_average(hyperspectral_image, mask)
    
    
    train_bbox, train_segm, train_labels, train_paths = load_data(train_annotation_path, train_image_dir)
    
    
    
    import pandas as pd
    
    bbox = train_bbox
    segm = train_segm
    label = train_labels
    
    dict = {"label": label, "bbox": train_bbox, "segmentation": train_segm, "label": train_labels}
    data = pd.DataFrame(dict)
    
    
    
    test_images, test_labels, test_ids = load_data(test_annotation_path, test_image_dir)

    PLS_train, PLS_test = PLS(X_train, y_train, X_test, y_test)


    #Split data
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Apply PLS for feature extraction
    n_components = 10
    pls_model = PLSRegression(n_components=n_components)
    pls_model.fit(X_train_scaled, Y_train)
    
    #Transform data using PLS model
    X_train_pls = pls_model.transform(X_train_scaled)
    X_test_pls = pls_model.transform(X_test_scaled)
    """
