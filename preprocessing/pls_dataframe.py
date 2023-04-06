
#### PLS 
from Display_mask import load_coco
from numpy_improved_kernel import PLS
import os
import numpy as np
#from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from watershed_v2 import preprocess_image, watershedd
from crop_from_mask import overlay_on_larger_image, fill_mask, crop_from_mask
from watershed_2_coco import watershed_2_coco
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import compress


def spectral_test(img_path):
    np.seterr(divide = 'ignore') 
    array = np.load(img_path)

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
        new_img[:,:,i-1]=(img_discarded[:,:,i*2-2]+img_discarded[:,:,(i*2-1)])/2

    return new_img


def pixel_average(hyperspectral_image, mask):
    # Calculate the pixel average for each unique label in the mask
    unique_labels = np.unique(mask) # Two labels, being background and foreground (0,1)
    pixel_averages = []

    
    for label in unique_labels:
        if (label == 0) or (label == 300):  # Assuming 0 is the background label
            continue # Do not take average which includes background
        
        else:
            
            grain_mask = (mask == label)
            grain_pixels = hyperspectral_image[grain_mask, :]
            grain_pixel_average = np.mean(grain_pixels, axis=0)
            pixel_averages.append(grain_pixel_average)

    return pixel_averages


def checkicheck(dataset, pseudo_image_dir, hyperspectral_path, training=True):
    
    
    hyper_path = []
    pseudo_rgbs = []
    img_name = []
    
    #hypersp_imgs = [] 
    
    # Looping over all images with their corresponding image_id
    for image_id in range(len(dataset["images"])):
        #Loading path to pseudo_rgb image
        image_name = dataset["images"][image_id]["file_name"]
        pseudo_rgb = os.path.join(pseudo_image_dir, image_name)
        
        
        # As the coco-dataset has initial-index 0, we start by this
        image_id += 1
        
        # Retrieving folder of the hyperspectral images
        hyper_folder = os.path.join(hyperspectral_path, "_".join(pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0].split("_")[1:3]))
        
        # Extracting the hyperspectral files
        hyperspectral_imgs = os.listdir(hyper_folder)
        # Making sure we are not extracting the multiplied- and subtracted-files
        hyperspectral_imgs = [i for i in hyperspectral_imgs if (("Multiplied" not in i) and ("subtracted" not in i))]
        
        if training:
            # Only retrieving the specific file-name of the pseudo-rgb - used to compare with the hyperspectral image-name
            pseudo_name = pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][9+len(hyper_folder.split("\\")[-1])+1:] + ".npy"
        
        else:
            # Only retrieving the specific file-name of the pseudo-rgb - used to compare with the hyperspectral image-name
            pseudo_name = pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][5+len(hyper_folder.split("\\")[-1])+1:] + ".npy"
            
        if pseudo_name in hyperspectral_imgs:
            
            [hyper_path.append(os.path.join(hyper_folder, i)) for i in hyperspectral_imgs if pseudo_name in i]
            pseudo_rgbs.append(pseudo_rgb)
            img_name.append(image_name)
        else:
            continue
    return hyper_path, pseudo_rgbs, img_name
    

def create_dataframe(hyperspectral_img, pseudo_rgb, image_name):
    
    # Creating dataframe with 103 rows - 1 label row and 102 for the channels
    df = pd.DataFrame( columns = [None]*103)
    df.columns = [f"wl{i}" for i in range(0, 103)]
    
    for i in range(len(image_name)):
        
        # Retrieving shape of pseudo_rgb - shall be used in creation of background
        pseudo_shape = cv.imread(pseudo_rgb[i]).shape
        background = np.zeros(pseudo_shape,dtype = np.uint8)
        background = cv.cvtColor(background, cv.COLOR_BGR2RGB)
        
        
        
        ##Now we have the full path to hyperspectral image
        # Load spectral image correctly
        spectral_img = spectral_test(hyperspectral_img[i])
        
        # Gain binary mask from pseudo_rgb through Otsu's method
        _, mask_h = preprocess_image(pseudo_rgb[i])
        
        # Pixel_average of spectral_img based on the binary mask
        pixel_avg = pixel_average(spectral_img, mask_h)
        
        
        ##At last we need one-hot-encoded labels
        # One-Hot-Encoded Labels
        class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
        
        labels = [int(names in image_name[i]) for names in class_list]
        
        # Creating list of len 103 with specific one-hot label and respective 102 pixel-averages
        temp = [labels] + pixel_avg[0].tolist()
        
        # Appending this to the dataframe
        df.loc[len(df)] = temp
    
    # Saving dataframe for later use
    df.to_csv("Pixel_avg_dataframe.csv",index=False)

        
    return df


def PLS_classify(dataframe, pseudo_image_path, hyperspectral_img_path,pseudo_name):
    
    y = dataframe['wl0']
    X = dataframe.values[:, 1:]

    if train:
        Y = [y[i] for i in range(len(y))]
    else:
        Y = [eval(y[i]) for i in range(len(y))] # cause read_csv(), dont ask me why
    
    wl = np.arange(900, 1700, (1700-900)/102)
    

    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]

    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]
    with plt.style.context('ggplot'):
        for i in range(len(Y)):
            plt.plot(wl, X[i].T, color=list(compress(color, Y[i]))[0], label=list(compress(class_list, Y[i]))[0])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        plt.legend(by_label.values(), by_label.keys())
        plt.xlabel("Wavelengths (nm)")
        plt.ylabel("Absorbance")
        plt.savefig("absorbance", dpi=200)
        plt.show()
        

    classifier = PLS(algorithm=2)
    Y = np.array(Y)
    classifier.fit(X, Y, 102)        
    
    for pseudo_img, hyp_img, nam in zip(pseudo_rgb, hyper_folder,pseudo_name):
        #img_name = r"C:\Users\admin\Downloads\hyper\Training\Rye_Midsummer\Sparse_Series1_20_09_08_07_47_28.npy"

        # Load spectral image correctly
        spectral_img = spectral_test(hyp_img)
    
        
        im, img_t = preprocess_image(pseudo_img)
        labels, markers = watershedd(im, img_t)
        unique_labels = np.unique(markers) # Getting unique labels
        
        classify_img = markers.copy()
        plt.figure(dpi=200)
        plt.imshow(im)
        for mask_id in np.add(unique_labels,300)[1:]: # offsetting labels to avoid error if mask_id == 255
            mask = markers.copy()
            mask = np.add(mask,300)
            mask[mask != mask_id] = 0
            mask[mask == mask_id] = 255
            
            pixel_avg = pixel_average(spectral_img, mask)[0]
            result = classifier.predict(pixel_avg, A=17)
        
            #heyo = [np.argmax(result[i,j,:]) for i in range(len(result)) for j in range(len(result[i]))]
            #classification = np.reshape(heyo, im.shape[0:2])
            
            
            
            cropped_im = cv.bitwise_and(im, im, mask=np.uint8(mask[mask==mask_id]))
            
            
            
            contours, _ = cv.findContours(np.uint8(mask),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE to avoid RLE
            
            anno = watershed_2_coco(contours)
                            
     
            start_x = min(anno[0::2])
            start_y = min(anno[1::2])
            end_x = max(anno[0::2])-start_x
            end_y = max(anno[1::2])-start_y

            cropped = cropped_im[start_y:start_y+end_y,start_x:start_x+end_x]
            
            masking = overlay_on_larger_image(im,cropped)
            
            x, y = anno[0::2],anno[1::2] # comes in pair of [x,y,x,y,x,y], there split with even and uneven
            plt.fill(x, y,alpha=.3, color=color[np.argmax(result)],label = class_list[np.argmax(result)])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc="center left", bbox_to_anchor =(1,0.5))
        plt.axis("off")
        plt.title(nam[:-30])
        plt.show()
            

        
        


if __name__ == "__main__":
    
    # Training_data
    #annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
    train_annotation_path = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Training/COCO_Training.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    train_image_dir = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Training/images/"
    
    # Test_data
    test_annotation_path = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Training/COCO_Training.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    test_image_dir = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/images/"
    
    # Loading training/test dataset
    dataset_train = load_coco(train_annotation_path)
    dataset_test = load_coco(test_annotation_path)
    
    
    #Loading path to hyperspectral image
    hyperspectral_path_train = r"C:\Users\Cornelius\Downloads\e4Wr5LFI4L\Training"
    hyperspectral_path_test = r"C:\Users\Cornelius\Downloads\e4Wr5LFI4L\Test"

    
    
    train = False
    if train:
        hyper_folder, pseudo_rgb, pseudo_name = checkicheck(dataset_train, train_image_dir, hyperspectral_path_train)
        df_train = create_dataframe(hyper_folder, pseudo_rgb, pseudo_name)
    else:
        hyper_folder, pseudo_rgb, pseudo_name = checkicheck(dataset_test, test_image_dir, hyperspectral_path_test, training=False)
        df_train = pd.read_csv("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/Pixel_avg_dataframe.csv")
        #df_train = df_train.iloc[:,1:]
    PLS_classify(df_train, pseudo_rgb, hyper_folder,pseudo_name)
    

    
    