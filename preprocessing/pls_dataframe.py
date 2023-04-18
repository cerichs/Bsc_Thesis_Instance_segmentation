
#### PLS 
from Display_mask import load_coco
from numpy_improved_kernel import PLS
import os
import numpy as np
#from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from watershed_v2 import preprocess_image, watershedd
from crop_from_mask import overlay_on_larger_image, fill_mask, crop_from_mask
from watershed_2_coco import watershed_2_coco, empty_dict, export_json
from simple_object_placer import coco_next_anno_id,coco_next_img_id
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.draw import polygon
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
        #hyper_folder = os.path.join(hyperspectral_path, "_".join(pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0].split("_")[1:3]))
        
        # Extracting the hyperspectral files
        hyperspectral_imgs = os.listdir(hyperspectral_path)
        # Making sure we are not extracting the multiplied- and subtracted-files
        hyperspectral_imgs = [i for i in hyperspectral_imgs if (("Multiplied" not in i) and ("subtracted" not in i))]
        
        if training:
            # Only retrieving the specific file-name of the pseudo-rgb - used to compare with the hyperspectral image-name
            #pseudo_name = pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][9+len(hyper_folder.split("\\")[-1])+1:] + ".npy"
            pseudo_name = pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0] + ".npy"
        else:
            # Only retrieving the specific file-name of the pseudo-rgb - used to compare with the hyperspectral image-name
            #pseudo_name = pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0][5+len(hyper_folder.split("\\")[-1])+1:] + ".npy"
            pseudo_name = pseudo_rgb.split("\\")[-1].split("/")[-1].split(".")[0] + ".npy"
            
            
        if pseudo_name in hyperspectral_imgs:
            
            [hyper_path.append(os.path.join(hyperspectral_path, i)) for i in hyperspectral_imgs if pseudo_name in i]
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
    df.to_csv("Pixel_avg_dataframe_test.csv",index=False)

        
    return df

def mean_centering_masks(pixel_avg_mask, ref = None):
    #print(pixel_avg_mask.shape)
    
    if ref is None: 
        ref = np.mean(pixel_avg_mask, axis=0)
    else:
        ref = ref
    
        
    for i,value in enumerate(pixel_avg_mask):
        pixel_avg_mask[i] = value - ref[i]
        
    
    ref = list(ref)
    hyp = list(pixel_avg_mask)
    fit = np.polyfit(ref, hyp, 1, full=True)
    # Apply correction
    data_msc = (pixel_avg_mask - fit[0][1]) / fit[0][0]
    return data_msc




def msc_hyp(hyperspectral_dataframe, ref = None):
    
    
    if ref is None:
        #Get the reference spectrum. Estimate it from the mean    
        ref = np.mean(hyperspectral_dataframe, axis=0)
    else:
        ref = ref
    
    # Define a new array and populate it with the data    
    data_msc = np.zeros_like(hyperspectral_dataframe)
    for i in range(hyperspectral_dataframe.shape[0]):
        # Run regression
        ref = list(ref)
        hyp = list(hyperspectral_dataframe[i,:])
        fit = np.polyfit(ref, hyp, 1, full=True)
        # Apply correction
        data_msc[i,:] = (hyperspectral_dataframe[i,:] - fit[0][1]) / fit[0][0] 
        
    return data_msc, ref
    
    

def mean_centering(data, ref = None):
    
    # Check if data already is MSC-processed
    if isinstance(data, (np.ndarray, np.generic)):
        mean_list = np.mean(data, axis=0)
        data_mean = data - mean_list if ref is None else data - ref
        return data_mean, mean_list
    
    # Check if data is still original dataframe-type
    elif isinstance(data, pd.DataFrame):
        mean_list = data.mean(axis=0)
        data_mean = data - mean_list if ref is None else data - ref
        return data_mean, mean_list
            
        
 
    
def add_2_coco(dict_coco,dataset,annotations,pseudo_img,class_id):
    if "Test" in pseudo_img or "Training" in pseudo_img:
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
    else:
        return


def PLS_show(classifier, type_classifier, pseudo_rgb, hyper_folder, pseudo_name, dataset, class_list, color, mean_list = None):
    
    test_list = []
    #df_sanity = pd.read_csv("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/Pixel_avg_dataframe_test.csv")
    df_sanity = pd.read_csv(r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\Pixel_avg_dataframe_test.csv")
    corrected, _ = mean_centering(df_sanity.values[:,1:], ref = mean_list)
    XXX =[list(i) for i in corrected]

    dict_coco = empty_dict()
    dict_coco["categories"] = dataset["categories"]
    count = 0
    
    for k, (pseudo_img, hyp_img, nam) in enumerate(zip(pseudo_rgb, hyper_folder,pseudo_name)):
        #img_name = r"C:\Users\admin\Downloads\hyper\Training\Rye_Midsummer\Sparse_Series1_20_09_08_07_47_28.npy"

        # Load spectral image correctly
        spectral_img = spectral_test(hyp_img)
    
        im, img_t = preprocess_image(pseudo_img)
        labels, markers = watershedd(im, img_t,plot=True)
        unique_labels = np.unique(markers) # Getting unique labels
        
        plt.figure(dpi=400)
        plt.imshow(im)
        result = classifier.predict(XXX[k],A=17)
        temp_name = "False"
        if class_list[np.argmax(result)] in nam:
            count+=1
            temp_name = "Correct"
        print(f"Image Classification, {type_classifier}: "+class_list[np.argmax(result)])
        print("Original image name: " +nam)
        
        spread = dict.fromkeys(class_list,0)
        for mask_id in np.add(unique_labels,300)[1:]: # offsetting labels to avoid error if mask_id == 255
            mask = markers.copy()
            mask = np.add(mask,300)
            mask[mask != mask_id] = 0
            mask[mask == mask_id] = 255
            
            pixel_avg = pixel_average(spectral_img, mask)[0]
            if mean_list is not None:
                pixel_avg = mean_centering_masks(pixel_avg, ref = mean_list)
            
            result = classifier.predict(pixel_avg, A=17)

            cropped_im = cv.bitwise_and(im, im, mask=np.uint8(mask[mask==mask_id]))
            
            contours, _ = cv.findContours(np.uint8(mask),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE to avoid RLE
            if (len(np.squeeze(contours))) > 2:

                anno = watershed_2_coco(contours)
                
                start_x = min(anno[0::2])
                start_y = min(anno[1::2])
                end_x = max(anno[0::2])-start_x
                end_y = max(anno[1::2])-start_y
    
                cropped = cropped_im[start_y:start_y+end_y,start_x:start_x+end_x]
                
                masking = overlay_on_larger_image(im,cropped)
                
                x, y = anno[0::2],anno[1::2] # comes in pair of [x,y,x,y,x,y], there split with even and uneven
                plt.fill(x, y,alpha=.3, color=color[np.argmax(result)],label = class_list[np.argmax(result)])
                
                dict_coco = add_2_coco(dict_coco,dataset,anno,pseudo_img,np.argmax(result))
                spread[class_list[np.argmax(result)]]+=1
            #except:
             #   print("Warning: Skipping object, Watershed gave 1 pixel object") # it sometimes predict 1 pixel instead of polygon
        dict_coco['images'].append({'id':coco_next_img_id(dict_coco),
                            'file_name': f"{nam}.jpg",
                            'license':1,
                            'height':im.shape[0],
                            'width':im.shape[1]})
        
        print(spread)
        print("_____________")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),loc="center left", bbox_to_anchor =(1,0.5))
        plt.axis("off")
        plt.title(f"{nam[:-30]}, {type_classifier}")
        plt.show()
    print(count)  
    export_json(dict_coco,"PLS_coco.json")


def PLS_classify(dataframe, pseudo_image_path, hyperspectral_img_path,pseudo_name,dataset, plot=False):
    
    # Creating three instances to perform PLS
        #1. Original data
        #2. Mean-centered data
        #3. MSC-mean-centered data
        
    #1. Original data
    y = dataframe['wl0']
    X = dataframe.values[:, 1:]
    
    #2. Mean-centered data
    Xmean, Xmean_mean_list = mean_centering(X)
    
    #3. MSC-Mean-centered data
    Xmsc, mean_list = msc_hyp(X) #First, peforms MSC
    Xmsc_mean, Xmsc_mean_list = mean_centering(Xmsc) #Then, mean-center the MSC-data
    
    
    
    if train:
        Y = [y[i] for i in range(len(y))]
    else:
        Y = [eval(y[i]) for i in range(len(y))] # cause read_csv(), dont ask me why
    
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]

    # Plots the absorbance of each instance
    if plot:
        wl = np.arange(900, 1700, (1700-900)/102)
        with plt.style.context('ggplot'):
            for i in range(len(Y)):
                plt.plot(wl, X[i].T, color=list(compress(color, Y[i]))[0], label=list(compress(class_list, Y[i]))[0])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            
            plt.legend(by_label.values(), by_label.keys())
            plt.xlabel("Wavelengths (nm)")
            plt.ylabel("Absorbance")
            plt.title("Original Data", loc='center')
            plt.savefig("absorbance_original", dpi=400)
            plt.show()
            
            for i in range(len(Y)):
                plt.plot(wl, Xmean[i].T, color=list(compress(color, Y[i]))[0], label=list(compress(class_list, Y[i]))[0])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            
            plt.legend(by_label.values(), by_label.keys())
            plt.xlabel("Wavelengths (nm)")
            plt.ylabel("Absorbance")
            plt.title("Mean-centered", loc='center')
            plt.savefig("absorbance_mean", dpi=400)
            plt.show()
            
            for i in range(len(Y)):
                plt.plot(wl, Xmsc_mean[i].T, color=list(compress(color, Y[i]))[0], label=list(compress(class_list, Y[i]))[0])
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            
            plt.legend(by_label.values(), by_label.keys())
            plt.xlabel("Wavelengths (nm)")
            plt.ylabel("Absorbance")
            plt.title("Mean-centered-MSC", loc='center')
            plt.savefig("absorbance_msc", dpi=400)
            plt.show()
        

    classifier_orig = PLS(algorithm=2)
    classifier_mean = PLS(algorithm=2)
    classifier_mscmean = PLS(algorithm=2)
    Y = np.array(Y)
    
    classifier_orig.fit(X, Y, 102)
    classifier_mean.fit(Xmean, Y, 102) 
    classifier_mscmean.fit(Xmsc_mean, Y, 102)
    
    PLS_show(classifier_orig, "Original", pseudo_image_path, hyperspectral_img_path, pseudo_name, dataset, class_list, color, mean_list = None)
    PLS_show(classifier_mean, "Mean-Centered", pseudo_image_path, hyperspectral_img_path, pseudo_name, dataset, class_list, color, Xmean_mean_list)
    PLS_show(classifier_mscmean, "MSC-Mean-Centered", pseudo_image_path, hyperspectral_img_path, pseudo_name, dataset, class_list, color, Xmsc_mean_list)

       
        
        


if __name__ == "__main__":
    
    # Training_data
    #annotation_path = r'C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_Test.json'
    
    #train_annotation_path = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Training/COCO_Training.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    #train_image_dir = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Training/images/"
    train_annotation_path =r"C:\Users\admin\Downloads\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\COCO_Training.json"
    train_image_dir = r"C:\Users\admin\Downloads\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\images"

    # Test_data
    #test_annotation_path = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/COCO_Test.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    #test_image_dir = "C:/Users/Cornelius/Downloads/DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo/Test/images/"
    test_annotation_path = r"C:\Users\admin\Downloads\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\COCO_Test.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    test_image_dir = r"C:\Users\admin\Downloads\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\images"
    
    # Loading training/test dataset
    dataset_train = load_coco(train_annotation_path)
    dataset_test = load_coco(test_annotation_path)
    
    
    #Loading path to hyperspectral image
    #hyperspectral_path_train = r"C:\Users\Cornelius\Downloads\e4Wr5LFI4L\Training"
    hyperspectral_path_train = r"C:\Users\admin\Downloads\hyper"
    
    #hyperspectral_path_test = r"C:\Users\Cornelius\Downloads\e4Wr5LFI4L\Test"

    
    
    train = True
    if train:
        #hyper_folder, pseudo_rgb, pseudo_name = checkicheck(dataset_test, test_image_dir, hyperspectral_path_test,training=False)
        hyper_folder, pseudo_rgb, pseudo_name = checkicheck(dataset_train, train_image_dir, hyperspectral_path_train)
        df_train = create_dataframe(hyper_folder, pseudo_rgb, pseudo_name)
        dataset = dataset_train
    else:
        hyper_folder, pseudo_rgb, pseudo_name = checkicheck(dataset_test, test_image_dir, hyperspectral_path_test, training=False)
        df_train = pd.read_csv("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/Pixel_avg_dataframe_test.csv")
        dataset = dataset_test
    PLS_classify(df_train, pseudo_rgb, hyper_folder, pseudo_name, dataset, plot=True)
    

    
    