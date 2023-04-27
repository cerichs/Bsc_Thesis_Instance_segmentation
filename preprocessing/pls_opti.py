from Display_mask import load_coco
from numpy_improved_kernel import PLS
import os
import numpy as np
from watershed_2_coco import watershed_2_coco, empty_dict, export_json
from sklearn.metrics import mean_squared_error
from watershed_v2 import preprocess_image, watershedd
from simple_object_placer import coco_next_anno_id,coco_next_img_id
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon
from itertools import compress
import cv2 as cv
from tqdm import tqdm


def find_imgs(dataset, hyp_folder, pseudo_folder):
    
    hyper_path = []
    pseudo_rgbs = []
    img_name = []
    ids = []
    
    for image_id in range(len(dataset["images"])):
       
        # Extracting file-names in dataset
        image_names = dataset["images"][image_id]["file_name"]
        
        # Extracting paths to pseudo-rgb-images
        pseudo_rgb_path = os.path.join(pseudo_folder, image_names)
        
        # Extracting paths to hyperspectral-images
        hyperspectral_imgs = os.listdir(hyp_folder)
        
        # Filter out any multiplied or subtracted hyperspectral images
        hyperspectral_imgs = [i for i in hyperspectral_imgs if ( ("Multiplied" not in i) and ("subtracted" not in i) )]
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
    return hyper_path, pseudo_rgbs, img_name, ids


def process_data(dataset, hyper_imgs, pseudo_imgs, img_names, image_ids, whole_img=False):
    
    HSI = [spectral_test(hyper_imgs[image]) for image in range(len(hyper_imgs))]
    g_masks = []
    
    for i in range(len(hyper_imgs)):
        if whole_img:
            _, grains_mask = preprocess_image(pseudo_imgs[i])
            g_masks.append([grains_mask])
            split = "whole_img"
        else:
            grains_mask, name, image_id = extract_binary_kernel(dataset, pseudo_imgs[i], img_names[i])
            g_masks.append(grains_mask)  
            split = "grain"
            
    print(f"g_masks   :  {len(g_masks)}")
    print(f"g_masks[0]   :  {len(g_masks[0])}")
    ids = []
    X = []
    y = []
    
    for images in range(len(g_masks)):
        print(f"len(g_masks[images])  :  {len(g_masks[images])}")
        print(f"Currently processing pixel-average on image-number :  {images+1}  out of {len(g_masks)}")
        
        pixel_avg, label, img_id = pixel_average(HSI[images], g_masks[images], image_ids[images], img_names[images])
        ids.append(img_id)
        X.append(pixel_avg)
        y.append(label)
    return ids, X, y, HSI, split


def spectral_test(img_path):
    """
    Perform spectral transformation on the given hyperspectral image.
    
    Parameters:
    img_path (str): Path to the hyperspectral image file.
    
    Returns:
    new_img (ndarray): Spectrally transformed hyperspectral image.
    """
    
    # Check if the image is a validation or training image
    np.seterr(divide='ignore')
    array = np.load(img_path)
    
    # Get the original filename of the image, and the corresponding subtracted and multiplied filenames
    hyp_orig = img_path.split("\\")[-1]
    temp = ("_").join(img_path.split("_")[3:])
    sub = "subtracted_" + temp
    mult = "Multiplied_" + temp
        
        
    # Load the subtracted and multiplied files
    sub = np.load(os.path.join(os.path.split(img_path)[0], sub))
    mult = np.load(os.path.join(os.path.split(img_path)[0], mult))
    
    # Perform the spectral transformation on the image
    array = (array / mult) + sub
    temp = -np.log10(array)
    if np.sum(np.isinf(temp)):
        temp[np.isinf(temp)] = 0
        
    img_discarded = temp[:, :, 9:213]
    new_img = np.zeros((img_discarded.shape[0], img_discarded.shape[1], int(img_discarded.shape[2] / 2)))
    for i in range(1, (int(len(img_discarded[0, 0, :]) / 2) + 1), 1):
        new_img[:, :, i - 1] = (img_discarded[:, :, i * 2 - 2] + img_discarded[:, :, (i * 2 - 1)]) / 2

    return new_img


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
    im, mask = preprocess_image(pseudo_rgbs_path)
    
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
    for mask in tqdm(binary_mask):
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


def create_dataframe(image_ids, labels, pixel_averages, split):
    """
    Create a pandas dataframe containing the average pixel values for each grain mask in the given images,
    along with the corresponding image IDs and one-hot-encoded labels.
    
    Parameters:
    image_ids (list): A list of image IDs.
    labels (list): A list of one-hot-encoded labels for each grain mask in the images.
    pixel_averages (list): A list of average pixel values for each grain mask in the images.
    split (str): A string indicating whether the data is for training, validation, or testing.
    
    Returns:
    df (DataFrame): A pandas dataframe containing the average pixel values for each grain mask in the images,
    along with the corresponding image IDs and one-hot-encoded labels.
    """
    
    # Creating dataframe with 103 columns - 1 label and 102 for the channels
    df = pd.DataFrame(columns=[None]*104)
    df.columns = ["image_id"] + ["label"] + [f"wl{i}" for i in range(1, 103)]
    
    for image in range(len(image_ids)):
        for mask in range(len(image_ids[image])):
            # Creating list of length 104 with image_id, one-hot label, and respective 102 pixel-averages
            temp = [image_ids[image][mask]] + [labels[image][mask]] + pixel_averages[image][mask].tolist()
        
            # Appending this to the dataframe
            df.loc[len(df)] = temp
            
    # Saving dataframe for later use
    df.to_csv(f"Pixel_grain_avg_dataframe_{split}.csv", index=False)
        
    return df


def mean_centering(data, ref = None):
    
    mean = []
    
    mean_list = data.mean(axis=0)
    data_mean = [data - mean_list if ref is None else data - ref]
    print(data_mean)
    for i in range(len(data)):
        mean.append( np.array(data_mean[0].loc[i].tolist()) )
    
    return pd.DataFrame(mean)

def msc_hyp(hyperspectral_dataframe, ref = None):
    
    
    if ref is None:
        # Get the reference spectrum. Estimate it from the mean. This is computed during training    
        ref = hyperspectral_dataframe.mean(axis=0)
    else:
        # If reference is not None, we are running on test-data
        ref = ref
    
    # Define a new array and populate it with the data    
    data_msc = np.zeros_like(hyperspectral_dataframe)
    for i in range(hyperspectral_dataframe.shape[0]):
        ref = list(ref)
        hyp = list(hyperspectral_dataframe.iloc[i,:])
        
        # Run regression
        print(len(ref))
        print(len(hyp))
        fit = np.polyfit(ref, hyp, 1, full=True)

        # Apply correction
        data_msc[i,:] = (hyp - fit[0][1]) / fit[0][0]
        
        
    return pd.DataFrame(data_msc), ref


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
        
        # Plot the legend with sorted class names
        plt.legend(sorted_handles, sorted_labels, loc=3)
        plt.xlabel("Wavelengths (nm)")
        plt.ylabel("Absorbance")
        plt.title(f"{type_classifier} Data", loc='center')
        plt.savefig(f"pls_results/absorbance_{type_classifier}.png", dpi=400)
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
        print(len(X))
        print(len(Y))
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
    plt.savefig(f"pls_results/RMSE_{type_classifier}.png", dpi=400)
    plt.show()

    # Visualization of results
    plt.plot(range(1, 103), np.array(accuracy), '-o', c="r", markersize=2)
    plt.xlabel('Components')
    plt.ylabel('Accuracy')
    plt.title(f'argmax(accuracy)={optimal_accur} for {type_classifier}')
    plt.savefig(f"pls_results/Accuracy_{type_classifier}.png", dpi=400)
    plt.show()

    return optimal_accur, optimal_RMSE, classifier

def add_2_coco(dict_coco,dataset,annotations,pseudo_img,class_id):
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
        
    test_list = []
    #df_sanity = pd.read_csv("C:/Users/Cornelius/Documents/GitHub/Bscproject/Bsc_Thesis_Instance_segmentation/preprocessing/Pixel_avg_dataframe_test.csv")
    df_sanity = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation-main\preprocessing\Pixel_grain_avg_dataframe_test_whole_img.csv")
    corrected = df_sanity.values[:,2:]
    XXX =[list(i) for i in corrected]
    
    
    if type_classifier is None:
        type_classifier = "Original"
            
    dict_coco = empty_dict()
    dict_coco["categories"] = dataset["categories"]
    count = 0

    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]

    for image in range(len(img_names)):
        spectral_img = HSI[image]
        
        
        rgb_image, img_tres = preprocess_image(img_path[image])
        labels, markers = watershedd(rgb_image, img_tres, plot=False)

        # Perform watershed segmentation on each of pseudo-RGB image-mask
        #im, img_t = preprocess_image(pseudo_img)
        
        "_, markers = watershedd(0, masks[image][binary_mask])"

        plt.figure(dpi=400)
        plt.imshow(rgb_image)
        
        unique_labels = np.unique(markers)
        
                
        if not train:
            # Get prediction for the entire pseudo-RGB image
            result1 = classifier.predict(XXX[image], A=5)
    
            temp_name = "False"
            if class_list[np.argmax(result1)] in img_names[image]:
                count+=1
            print(f"Image Classification, {type_classifier}: " + class_list[np.argmax(result1)])
            print("Original image name: " + img_names[image])

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
        plt.savefig(f"pls_results/{type_classifier}_{img_names[image]}.png", dpi=400)
        plt.show()
    print(count)  
    export_json(dict_coco,f"pls_results/PLS_coco_{type_classifier}.json")      
    
    
    
def main():
    """
    Main function that loads the dataset and performs PLS classification on the training and test sets.
    """
    # Load paths
    train_annotation_path =r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\COCO_Training.json"
    train_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Training\images"
    #test_annotation_path = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\COCO_Test.json"
    #test_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Test\images"
    test_annotation_path = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\COCO_Validation.json"
    test_image_dir = r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\images"
    
    hyperspectral_path_train = r"C:\Users\jver\Desktop\Training"
    #hyperspectral_path_test = r"C:\Users\jver\Desktop\Test"
    hyperspectral_path_test = r"C:\Users\jver\Desktop\Validation"
    
    # Load training and test datasets
    train_dataset = load_coco(train_annotation_path)
    test_dataset = load_coco(test_annotation_path)


    # Train PLS classifier
    train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids = find_imgs(train_dataset, hyperspectral_path_train, train_image_dir)

    
    image_ids, pixel_averages, labels, HSI_train, split = process_data(train_dataset, train_hyper_imgs, train_pseudo_imgs, train_img_names, train_image_ids, whole_img=True)
    train_dataframe = create_dataframe(image_ids, labels, pixel_averages, split=f"train_{split}")
    
    X_train = train_dataframe.iloc[:,2:]
    y_train = train_dataframe.label
    
    
    mean_data = mean_centering(X_train)
    msc_data, ref_train = msc_hyp(X_train, ref=None)
    msc_mean = mean_centering(msc_data, ref=None)
    
    spectra_plot(X_train, y_train, type_classifier=f"Train Original {split}")
    _, _, train_classifier = PLS_evaluation(X_train, y_train, type_classifier=f"Train Original {split}")
    PLS_show(train_classifier, X_train, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, type_classifier=f"Train Original {split}")
    
    
    spectra_plot(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    _, _, train_classifier_mean = PLS_evaluation(mean_data, y_train, type_classifier=f"Train Mean-Centered {split}")
    PLS_show(train_classifier_mean, mean_data, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, type_classifier=f"Train Mean-Centered {split}")
    
    
    spectra_plot(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    _ , _ , train_classifier_msc = PLS_evaluation(msc_mean, y_train, type_classifier=f"Train MSC Mean-Centered {split}")
    PLS_show(train_classifier_msc, msc_mean, y_train, HSI_train, 21, train_dataset, train_pseudo_imgs, train_img_names, type_classifier=f"Train MSC Mean-Centered {split}")
    
        
    # Test PLS classifier
    test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids = find_imgs(test_dataset, hyperspectral_path_test, test_image_dir)
    image_ids, pixel_averages, labels, HSI_test, split = process_data(test_dataset, test_hyper_imgs, test_pseudo_imgs, test_img_names, test_image_ids, whole_img=True)
    test_dataframe = create_dataframe(image_ids, labels, pixel_averages, split=f"test_{split}")
    
    X_test = test_dataframe.iloc[:,2:]
    y_test = test_dataframe.label
    
    
    mean_data = mean_centering(X_test, ref=ref_train)
    msc_data, _ = msc_hyp(X_test, ref=ref_train)
    msc_mean = mean_centering(msc_data, ref=ref_train)
    
    spectra_plot(X_test, y_test, type_classifier=f"Test Original {split}")
    _, RMSE, _ = PLS_evaluation(X_test, y_test, classifier=train_classifier, type_classifier=f"Test Original {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier, X_test, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, type_classifier=f"Test Original {split}", train=False)
    
    
    spectra_plot(mean_data, y_test, type_classifier=f"Test Mean-Centered {split}")
    _, RMSE, _ = PLS_evaluation(mean_data,  y_test, classifier=train_classifier_mean, type_classifier=f"Test Mean-Centered {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_mean, mean_data, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, type_classifier=f"Test Mean-Centered {split}", train=False)
    
    
    spectra_plot(msc_mean, y_test, type_classifier=f"Test MSC Mean-Centered {split}")
    _ , RMSE, _ = PLS_evaluation(msc_mean, y_test, classifier=train_classifier_msc, type_classifier=f"Test Mean-Centered {split}")
    print(f"RMSE-components : {RMSE}")
    PLS_show(train_classifier_msc, msc_mean, y_test, HSI_test, RMSE, test_dataset, test_pseudo_imgs, test_img_names, type_classifier=f"Test MSC Mean-Centered {split}", train=False)
    
    
    


    
    
    """


    #train_dataframe1 - whole image
    #train_dataframe - grains
    
    """




if __name__ == "__main__":
    main()



