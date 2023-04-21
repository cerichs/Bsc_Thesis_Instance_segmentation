
#### PLS 
from Display_mask import load_coco
from numpy_improved_kernel import PLS
import os
import numpy as np
#from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from watershed_v2 import preprocess_image, watershedd
from watershed_2_coco import watershed_2_coco, empty_dict, export_json
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.draw import polygon
from itertools import compress


def find_imgs(dataset, hyp_folder, pseudo_folder):
    
    hyper_path = []
    pseudo_rgbs = []
    img_name = []
    
    for image_id in range(len(dataset["images"])):
       
        # Extracting file-names in dataset
        image_names = dataset["images"][image_id]["file_name"]
        
        # Extracting paths to pseudo-rgb-images
        pseudo_rgb_path = os.path.join(pseudo_folder, image_names)
        
        # Extracting paths to hyperspectral-images
        hyperspectral_imgs = os.listdir(hyp_folder)
        # Not extracting the multiplied- and subtracted-files
        hyperspectral_imgs = [i for i in hyperspectral_imgs if ( ("Multiplied" not in i) and ("subtracted" not in i) )]
        #hyper_path = os.path.join(hyp_folder, image_name)

        # As the coco-dataset has initial-index 0, we start by this
        image_id += 1
        
        # Gaining pseudo-rgb-images with hyperspectral-extension to easier find linked files.
        pseudo_name = pseudo_rgb_path.split("\\")[-1].split("/")[-1].split(".")[0] + ".npy"
        
        # Finds the connecting names between pseudo-rgb- and hyperspectral-images
        if pseudo_name in hyperspectral_imgs:
            
            [hyper_path.append(os.path.join(hyp_folder, i)) for i in hyperspectral_imgs if pseudo_name in i]
            pseudo_rgbs.append(pseudo_rgb_path)
            img_name.append(pseudo_name.split(".")[0] + ".jpg")
        else:
            continue
    return hyper_path, pseudo_rgbs, img_name
    

def spectral_test(img_path):
    
    if "Validation" or "Train" in img_path:
        np.seterr(divide = 'ignore') 
        array = np.load(img_path)

        hyp_orig = img_path.split("\\")[-1]
        temp = ("_").join(img_path.split("_")[3:])
        sub = "subtracted_" + temp
        mult = "Multiplied_" + temp
        
    else:
            
        np.seterr(divide = 'ignore') 
        array = np.load(img_path)
    
        hyp_orig = img_path.split("\\")[-1].split(".")[0]
        sub = "subtracted_" + hyp_orig + ".npy"
        mult = "Multiplied_" + hyp_orig + ".npy" 
        
    """
    if "Train" in img_path:
        img_path = img_path
    else:
    """  
        
        
        
 
    sub = np.load( os.path.join(os.path.split(img_path)[0], sub) )
    mult = np.load( os.path.join(os.path.split(img_path)[0], mult) )
 
    array = (array / mult) + sub
    temp = -np.log10( array )
    img_discarded = temp[:,:,9:213]
    new_img = np.zeros(( img_discarded.shape[0], img_discarded.shape[1], int(img_discarded.shape[2] / 2) ))
    for i in range(1, (int(len(img_discarded[0,0,:])/2) + 1), 1):
        new_img[:,:,i-1] = ( img_discarded[:,:,i*2-2] + img_discarded[:,:,(i*2-1)] ) / 2

    return new_img



def extract_binary_kernel(dataset, pseudo_rgbs_path, img_name):
    
    grains = []
    name = []
    ids = []
    
    anno_img = dataset["images"]
    segment = dataset["annotations"]
    
    img_id = [anno_img[j]["id"] for j in range(len(anno_img)) if anno_img[j]["file_name"]==img_name]
    img_id = img_id[0]
    
    segmentation = [segment[k] for k in range(len(segment)) if segment[k]["image_id"]==img_id]
    
    im, mask = preprocess_image(pseudo_rgbs_path)


    grains = []
    name = []
    ids = []
    
    for annotation_numb in range(len(segmentation)):
        mini_img = np.zeros((mask.shape[0],mask.shape[1]),dtype=bool)
        x, y = (segmentation[annotation_numb]["segmentation"][0][0::2]),(segmentation[annotation_numb]["segmentation"][0][1::2])
        
        
        for x_x,y_y in zip(x,y):
            x_x, y_y = int(x_x), int(y_y)
            mini_img[y_y,x_x]=True
        binary_grain_mask=mini_img.astype(int)

        row, col = polygon(y, x, mini_img.shape)
        binary_grain_mask[row,col] = 1
        grains.append(binary_grain_mask)
        #print(img_name)
        name.append(img_name)
        ids.append(img_id)
   #print(ids)
    #image_id = [img_id]*len(grains)

        
    return grains, name, ids






def pixel_average(hyper_path, binary_mask, image_id, image_name):
    
    #grain_id = []
    pixel_averages = []
    one_hot = []
    #print(image_name)
    
    hyper_path = np.unique(hyper_path)
    #print(hyper_path)
    
    hyper_img = spectral_test(hyper_path[0])
    
    for i in range(len(binary_mask)):
        
        #hyper_img = [hyper_path[j] for j in range(len(hyper_path)) if image_names[i].split(".")[0] in hyper_path[j]]

        
        unique_labels = np.unique(binary_mask[i]) # Two labels, being background and foreground (0,1)
        
        for label in unique_labels:
            if (label == 0) or (label == 300):  # Assuming 0 is the background label
                continue # Do not take average which includes background
            
            else:
                # Retrieving both image_id and grain_id
                #image_ids.append(image_id[])
                #grainkernel_id = relevant2[annotation_numb]['id']
                #grain_id.append(grainkernel_id)
       
                grain_mask = (binary_mask[i] == label)
                #print(grain_mask.shape)
                # One-Hot-Encoded Labels
                class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
                
                labels = [int(names in image_name[i]) for names in class_list]
                one_hot.append(labels)
                
                
                
                grain_pixels = hyper_img[grain_mask, :]
                grain_pixel_average = np.mean(grain_pixels, axis=0)
                
                
                
                pixel_averages.append(grain_pixel_average)
   
    return one_hot, pixel_averages, image_id



def create_dataframe(image_ids, labels, pixel_averages, split):
    # Creating dataframe with 103 columns - 1 label and 102 for the channels
    df = pd.DataFrame( columns = [None]*104)
    df.columns = ["image_id"] + ["label"] + [f"wl{i}" for i in range(1, 103)]
    
    for i in range(len(image_ids)):
        for i, j, k in zip(image_ids[i], labels[i], pixel_averages[i]):
            # Creating list of len 104 with image_id, one-hot label and respective 102 pixel-averages
            temp = [i] + [j] + k.tolist()
        
            # Appending this to the dataframe
            df.loc[len(df)] = temp
            
        
    # Saving dataframe for later use
    df.to_csv(f"Pixel_grain_avg_dataframe_{split}.csv",index=False)
        
    return df


def spectra_plot(X, Y, type_classifier=None):
    
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]
    
    if type_classifier is None:
        type_classifier = "Original"
    
    wl = np.arange(900, 1700, (1700-900)/102)
    with plt.style.context('ggplot'):
        for i in range(len(Y)):
            for grain in range(len(Y[i])):
                plt.plot(wl, X[i][grain].T, color=list(compress(color, Y[i][grain]))[0], label=list(compress(class_list, Y[i][grain]))[0])
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        
        plt.legend(by_label.values(), by_label.keys())
        plt.xlabel("Wavelengths (nm)")
        plt.ylabel("Absorbance")
        plt.title(f"{type_classifier} Data", loc='center')
        plt.savefig("absorbance_{type_classifier}", dpi=400)
        plt.show()
    
def PLS_classify(dataframe, train=False):
    
    # Creating three instances to perform PLS
        #1. Original data
        #2. Mean-centered data
        #3. MSC-mean-centered data
      
    y = dataframe['label']
    X = dataframe.values[:, 2:]
    
    if train:
        Y = [y[i] for i in range(len(y))]
    else:
        Y = [eval(y[i]) for i in range(len(y))] # cause read_csv(), dont ask me why
        
    
    classifier = PLS(algorithm=2)
    #1. Original data
    #classifier_orig = PLS(algorithm=2)
    classifier.fit(X, Y, 102)
    
    #CV
    
    #cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)
    
    
    #loop to calculate error
    RMSE = []
    
    accuracy = []
    for i in range(1, 102+1):
        y_pred = classifier.predict(X, A=i)
        
        #print(np.argmax(y_pred, axis=1))
        #print(np.argmax(Y, axis=1))
        
        res = np.sum( ( np.argmax(y_pred, axis=1) ) == ( np.argmax(Y, axis=1) ) )
        #print(len(Y))
        
        accur = res / len(Y)
        
        accuracy.append(accur)
        
        
        
        #print(Y.shape)
        #score = np.sqrt(-1*cross_val_score(pls, X, Y, cv=cv_10, scoring='neg_mean_squared_error').mean())
        #count = 0
        #if Y == y_pred:
            #count += 1
        
        #accuracy = count / len(y_pred)
        
        
        score = np.sqrt(mean_squared_error(Y, y_pred))
        #print(Y, y_pred)
        RMSE.append(score)

    optimal_comp = [i for i in range(1, len(RMSE)) if (RMSE[i-1]-RMSE[i]) <10e-3]
    
    #Visualization of results
    plt.plot(range(1, 102+1), np.array(RMSE), '-v', c = "r")
    plt.xlabel('Components')
    plt.ylabel('RMSE')
    plt.title('Grains')
    plt.show()
    
    #Visualization of results
    plt.plot(range(1, 102+1), np.array(accuracy), '-v', c = "r")
    plt.xlabel('Components')
    plt.ylabel('Accuracy')
    plt.title('Grains')
    plt.show()
    
    return optimal_comp, classifier


def PLS_validation(optimal_comp, classifier, dataframe):
    
    
    y = dataframe['label']
    X = dataframe.values[:, 2:]
    
    
    Y = [y[i] for i in range(len(y))]
    #loop to calculate error
    RMSE = []
    
    accuracy = []
    for i in range(1, 102+1):
        y_pred = classifier.predict(X, A=i)
        
        #print(np.argmax(y_pred, axis=1))
        #print(np.argmax(Y, axis=1))
        
        res = np.sum( ( np.argmax(y_pred, axis=1) ) == ( np.argmax(Y, axis=1) ) )
        #print(len(Y))
        
        accur = res / len(Y)
        
        accuracy.append(accur)
        
        
        
        #print(Y.shape)
        #score = np.sqrt(-1*cross_val_score(pls, X, Y, cv=cv_10, scoring='neg_mean_squared_error').mean())
        #count = 0
        #if Y == y_pred:
            #count += 1
        
        #accuracy = count / len(y_pred)
        
        
        score = np.sqrt(mean_squared_error(Y, y_pred))
        #print(Y, y_pred)
        RMSE.append(score)

    optimal_comp = [i for i in range(1, len(RMSE)) if (RMSE[i-1]-RMSE[i]) <10e-3]
    
    #Visualization of results
    plt.plot(range(1, 102+1), np.array(RMSE), '-v', c = "r")
    plt.xlabel('Components')
    plt.ylabel('RMSE')
    plt.title('Grains')
    plt.show()
    
    #Visualization of results
    plt.plot(range(1, 102+1), np.array(accuracy), '-v', c = "r")
    plt.xlabel('Components')
    plt.ylabel('Accuracy')
    plt.title('Grains')
    plt.show()
    
    
    





def PLS_show(classifier, X, Y, type_classifier, pseudo_rgb, hyper_folder, pseudo_name, dataset, mean_list = None, plot=False):
    """
    Function that performs Partial Least Squares (PLS) classification on hyperspectral data and displays the results.
      
    Args:
    classifier: PLS classifier object
    X: Feature matrix (hyperspectral data)
    Y: Label vector
    type_classifier: String indicating the type of classifier (e.g., original, mean-centered, MSC-mean-centered)
    pseudo_rgb: List of pseudo-RGB images
    hyper_folder: List of paths to hyperspectral images
    pseudo_name: List of names for the pseudo-RGB images
    dataset: COCO dataset object
    mean_list: List of mean values to use for mean-centering
    plot: Boolean indicating whether or not to plot absorbance spectra for each instance
      
    Returns:
    None
    """
  
    test_list = []
    #df_sanity = pd.read_csv("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/Pixel_avg_dataframe_test.csv")
    df_sanity = pd.read_csv(r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\Pixel_avg_dataframe_test.csv")
    corrected, _ = mean_centering(df_sanity.values[:,1:], ref = mean_list)
    XXX =[list(i) for i in corrected]

    dict_coco = empty_dict()
    dict_coco["categories"] = dataset["categories"]
    count = 0
    
    
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
            plt.title(f"{type_classifier} Data", loc='center')
            plt.savefig("absorbance_{type_classifier}", dpi=400)
            plt.show()
             
    for k, (pseudo_img, hyp_img, nam) in enumerate(zip(pseudo_rgb, hyper_folder,pseudo_name)):
        #img_name = r"C:\Users\admin\Downloads\hyper\Training\Rye_Midsummer\Sparse_Series1_20_09_08_07_47_28.npy"
        
        # Load spectral image correctly
        #spectral_img = spectral_test(hyp_img)
        spectral_img = np.load(hyp_img)
    
        im, img_t = preprocess_image(pseudo_img)
        labels, markers = watershedd(im, img_t, plot=False)
        unique_labels = np.unique(markers) # Getting unique labels
        
        plt.figure(dpi=400)
        plt.imshow(im)
        result1 = classifier.predict(XXX[k],A=17)
        temp_name = "False"
        if class_list[np.argmax(result1)] in nam:
            count+=1
            temp_name = "Correct"
        print(f"Image Classification, {type_classifier}: " + class_list[np.argmax(result1)])
        print("Original image name: " +nam)
        spread = dict.fromkeys(class_list,0)
        for mask_id in np.add(unique_labels,300)[1:]: # offsetting labels to avoid error if mask_id == 255
            mask = markers.copy()
            mask = np.add(mask,300)
            mask[mask != mask_id] = 0
            mask[mask == mask_id] = 255
            
            pixel_avg = pixel_average(spectral_img, mask)[0]
            #if mean_list is not None:
            #    print(mean_list)
            #    pixel_avg = mean_centering_masks(pixel_avg, ref = mean_list)
            
            result2 = classifier.predict(pixel_avg, A=17)

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
                plt.fill(x, y, alpha=.3, color=color[np.argmax(result2)],label = class_list[np.argmax(result2)])
                
                dict_coco = add_2_coco(dict_coco,dataset,anno,pseudo_img,np.argmax(result2))
                spread[class_list[np.argmax(result2)]]+=1
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




if __name__ == "__main__":
    
    
    train_annotation_path =r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\COCO_Training.json"
    train_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\images"
    
    test_annotation_path = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\COCO_Validation.json"#image_dir = 'C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/'
    test_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\images"
    
    # Loading training/test dataset
    dataset_train = load_coco(train_annotation_path)
    dataset_test = load_coco(test_annotation_path)
    
    
    #Loading path to hyperspectral image
    #hyperspectral_path_train = r"C:\Users\Cornelius\Downloads\e4Wr5LFI4L\Training"
    hyperspectral_path_train = r"C:\Users\jver\Desktop\Training"
    
    hyperspectral_path_test = r"C:\Users\jver\Desktop\Validation"
    
    train = True
    
    if train:
        hyper_imgs, pseudo_imgs, img_names = find_imgs(dataset_train, hyperspectral_path_train, train_image_dir)
        
        ids = []
        X = []
        y = []
        for i,j,k in zip(hyper_imgs, pseudo_imgs, img_names):
            
            grains, name, image_id = extract_binary_kernel(dataset_train, j, k)
            
            label, grain_avg, _ = pixel_average(i, grains, ids, name)
            
            ids.append(image_id)
            X.append(grain_avg)
            y.append(label)
            
        
        dataframe = create_dataframe(ids, y, X, "train")
        spectra_plot(X, y)
        compos, classifier = PLS_classify(dataframe, train=True)
        train = False
        
    if not train:
        hyper_imgs, pseudo_imgs, img_names = find_imgs(dataset_test, hyperspectral_path_test, test_image_dir)
        
        print(hyper_imgs)
        ids = []
        X = []
        y = []
        for i,j,k in zip(hyper_imgs, pseudo_imgs, img_names):
            
            grains, name, image_id = extract_binary_kernel(dataset_test, j, k)
            
            label, grain_avg, _ = pixel_average(i, grains, ids, name)
            
            ids.append(image_id)
            X.append(grain_avg)
            y.append(label)
            
        
        dataframe = create_dataframe(ids, y, X, "test")
        spectra_plot(X, y)
        PLS_validation(compos, classifier, dataframe)
            
        
        
"""  
        image_id, grain_ids, labels, pixel_avg = grain_pixel_average(dataset_train, i, j, k)
        ids.append(image_id)
        X.append(pixel_avg)
        y.append(labels)
    
    df = create_dataframe(ids, y, X)
    PLS_classify(df, train=True)
    
    spectra_plot(X, y)
"""
    