import sys
sys.path.append("..")
from preprocessing.simple_object_placer import coco_next_img_id
from preprocessing.extract_process_grains import pixel_average
from preprocessing.preprocess_image import binarization
from .numpy_improved_kernel import PLS
from .watershed_2_coco import watershed_2_coco, empty_dict, export_json
from .watershed_v2 import watershedd
from .output_results import add_2_coco

import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from itertools import compress
import cv2 as cv


def spectra_plot(X, Y, type_classifier=None):
    """
    Plot the spectral curves for each grain in the images, with the legend sorted alphabetically by class name.
    
    Parameters:
    X (list): A list of pixel values for each grain in the images.
    Y (list): A list of one-hot-encoded labels for each grain in the images.
    type_classifier (str): A string indicating the type of data being plotted (training, validation, or testing).
    """

    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]
    
    if type_classifier is None:
        type_classifier = "Original"
    
    # Create an array of wavelengths for the x-axis
    wl = np.arange(900, 1700, (1700-900)/102)
    # Plot each image's/grain's spectral curve
    with plt.style.context('ggplot'):
        for label in range(len(Y)):

            # Set the line color based on the grain's class
            plt.plot(wl, X.loc[label].T, color=list(compress(color, Y.loc[label]))[0], label=list(compress(class_list, Y.loc[label]))[0])
    
        # Sort the legend alphabetically by class name
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        sorted_labels = sorted(by_label.keys())
        sorted_handles = [by_label[label] for label in sorted_labels]
        
        import os
        print(os.getcwd())
        # Plot the legend with sorted class names
        plt.legend(sorted_handles, sorted_labels, loc=3)
        plt.xlabel("Wavelengths (nm)")
        plt.ylabel("Absorbance")
        plt.title(f"{type_classifier} Data", loc='center')
        plt.savefig(f"two_stage/figures//absorbance_{type_classifier}.png", dpi=400)
        plt.show()


def PLS_evaluation(X, y, classifier=None, type_classifier=None):
    """
    Perform partial least squares (PLS) classification on the input dataframe.

    Parameters:
    -----------
    dataframe : pandas DataFrame
        The input dataframe containing the pixel grain averages.
    train : bool, optional (default=False)
        Whether to train the classifier or use it for prediction.

    Returns:
    --------
    optimal_comp : list
        A list of the optimal number of components to use for the classification.
    classifier : PLS instance
        The PLS classifier trained on the input dataframe.
    """

    Y = [y[i] for i in range(len(y))]

    if classifier is None:
        # Train the PLS classifier on the input data
        classifier = PLS(algorithm=2)
        classifier.fit(X, Y, 102)

    # Compute the RMSE and accuracy for different numbers of components
    RMSE = []
    accuracy = []
    for i in range(1, 103):
        y_pred = classifier.predict(X, A=i)
        res = (np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1)).sum()
        accuracy.append(res / len(Y))
        RMSE.append(np.sqrt(mean_squared_error(Y, y_pred)))


    # Find the optimal number of components based on the accuracy and RMSE
    optimal_accur = np.argmax(accuracy)
    optimal_RMSE = np.argmin(RMSE)

    # Visualization of results
    plt.plot(range(1, 103), np.array(RMSE), '-o', c="b", markersize=2)
    plt.xlabel('Components')
    plt.ylabel('RMSE')
    plt.title(f'argmin(RMSE)={optimal_RMSE} for {type_classifier}.png')
    plt.savefig(f"two_stage/figures/RMSE_{type_classifier}.png", dpi=400)
    plt.show()

    # Visualization of results
    plt.plot(range(1, 103), np.array(accuracy), '-o', c="r", markersize=2)
    plt.xlabel('Components')
    plt.ylabel('Accuracy')
    plt.title(f'argmax(accuracy)={optimal_accur} for {type_classifier}')
    plt.savefig(f"two_stage/figures/Accuracy_{type_classifier}.png", dpi=400)
    plt.show()

    return optimal_accur, optimal_RMSE, classifier



def PLS_show(classifier, X, Y, HSI, RMSE, dataset, img_path, img_names, type_classifier=None, train=True):
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
    
    r"""
    test_list = []
    #df_sanity = pd.read_csv("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/Pixel_avg_dataframe_test.csv")
    df_sanity = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation-main\preprocessing\Pixel_grain_avg_dataframe_test_whole_img.csv")
    corrected = df_sanity.values[:,2:]
    XXX =[list(i) for i in corrected]
    """
    
    if type_classifier is None:
        type_classifier = "Original"
            
    dict_coco = empty_dict()
    dict_coco["categories"] = dataset["categories"]
    count = 0

    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]

    for image in range(len(img_names)):
        spectral_img = HSI[image]
        
        
        rgb_image, img_tres = binarization(img_path[image])
        labels, markers = watershedd(rgb_image, img_tres, plot=False)

        # Perform watershed segmentation on each of pseudo-RGB image-mask
        #im, img_t = binarization(pseudo_img)
        
        "_, markers = watershedd(0, masks[image][binary_mask])"

        plt.figure(dpi=400)
        plt.imshow(rgb_image)
        
        unique_labels = np.unique(markers)
        
        """
        if not train:
            # Get prediction for the entire pseudo-RGB image
            result1 = classifier.predict(XXX[image], A=5)
    
            temp_name = "False"
            if class_list[np.argmax(result1)] in img_names[image]:
                count+=1
            print(f"Image Classification, {type_classifier}: " + class_list[np.argmax(result1)])
            print("Original image name: " + img_names[image])
        """
        
        # Create dictionary to keep track of predicted spread
        
        spread = dict.fromkeys(class_list, 0)
        
        for mask_id in np.add(unique_labels, 300)[1:]: # Offset labels to avoid error if mask_id == 255 from Watershed (happens if there are more than 255 grain-kernels)
            mask = markers.copy()
            mask = np.add(mask, 300)
            mask[mask != mask_id] = 0
            mask[mask == mask_id] = 255

            # Compute the pixel average of the spectral image for each grain_mask
            pixel_avg = pixel_average(spectral_img, [mask], None, img_names[image] )[0]

            # Get prediction for the mask
            result2 = classifier.predict(pixel_avg, A=RMSE)

            # Compute the cropped image for the mask
            cropped_im = cv.bitwise_and(rgb_image, rgb_image, mask=np.uint8(mask[mask==mask_id]))
            contours, _ = cv.findContours(np.uint8(mask), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE to avoid RLE
            if (len(np.squeeze(contours))) > 2:
                # Add the mask to the COCO dataset
                anno = watershed_2_coco(contours)
                dict_coco = add_2_coco(dict_coco, dataset, anno, img_names[image], np.argmax(result2))
    
                # Update the spread dictionary
                spread[class_list[np.argmax(result2)]] += 1
    
                # Compute the cropped image and overlay on the original image
                start_x = min(anno[0::2])
                start_y = min(anno[1::2])
                end_x = max(anno[0::2])-start_x
                end_y = max(anno[1::2])-start_y
    
                cropped = cropped_im[start_y:start_y+end_y,start_x:start_x+end_x]
                
                
                x, y = anno[0::2],anno[1::2] # comes in pair of [x,y,x,y,x,y], there split with even and uneven
                plt.fill(x, y, alpha=.3, color=color[np.argmax(result2)],label = class_list[np.argmax(result2)])
                
                dict_coco = add_2_coco(dict_coco, dataset, anno, img_names[image], np.argmax(result2))
                spread[class_list[np.argmax(result2)]]+=1
            #except:
            #   print("Warning: Skipping object, Watershed gave 1 pixel object") # it sometimes predict 1 pixel instead of polygon
        dict_coco['images'].append({'id':coco_next_img_id(dict_coco),
                            'file_name': f"{img_names[image]}.jpg",
                            'license':1,
                            'height': rgb_image.shape[0],
                            'width': rgb_image.shape[1]})
    
        print(spread)
        print("_____________")
        handles, labels = plt.gca().get_legend_handles_labels()
        
        #res = (np.argmax(result2, axis=1) == np.argmax(Y[image], axis=1)).sum()
        #accuracy = (res / len(Y[image]))
        
        # Sort the legend alphabetically by class name
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        sorted_labels = sorted(by_label.keys())
        sorted_handles = [by_label[label] for label in sorted_labels]
        
        plt.legend(sorted_handles, sorted_labels, loc="center left", bbox_to_anchor =(1,0.5))
        plt.axis("off")
        plt.title(f"{img_names[image][:-30]}, {type_classifier}")
        plt.savefig(f"two_stage/pls_results/{type_classifier}_{img_names[image]}.png", dpi=400)
        plt.show()
    print(count)  
    export_json(dict_coco,f"\two_stage/pls_results/PLS_coco_{type_classifier}.json")      


