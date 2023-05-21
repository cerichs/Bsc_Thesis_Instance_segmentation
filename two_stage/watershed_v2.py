import sys
sys.path.append("..")
from preprocessing.preprocess_image import binarization

import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import cv2 as cv
from matplotlib import pyplot as plt
#import os
import seaborn as sns
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches
    

def watershedd(original_im, image_t, minimum_distance=12, plot=False):
    # Finding foreground area
    # Finds the smallest distance between object and background
    dist_transform = cv.distanceTransform(np.uint8(image_t),cv.DIST_L2,5)
    
    l_max = peak_local_max(dist_transform, indices=False, min_distance=minimum_distance, labels=image_t)
    fg = np.int8(l_max)
    
    # Marker labelling
    ret, markers1 = cv.connectedComponentsWithAlgorithm(fg,connectivity=4,ccltype=cv.CCL_DEFAULT,ltype=cv.CV_32S )
    
    #print(ret)
    #print("")
    #print(markers1)
    markers = watershed(-dist_transform, markers1, mask=image_t)
    
    if plot:
        ## Dist_transform and pixel-plot
        #creating colors
        cvals = [-i for i in range(int(np.max(dist_transform)))][::-1][::2][:7]
        colors = ["midnightblue", "navy", "darkblue", "mediumblue", "royalblue", "cyan", "black"][::-1]
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)
        
        #creating subwindow of image (pretty centered - but one can change x_idx and y_idx)
        window_size = (32, 32)
        window_height, window_width = window_size
        x_idx = 232
        y_idx = 180
        top_left_x =  x_idx - window_width
        top_left_y =  y_idx - window_height
        bottom_right_x = top_left_x + window_width
        bottom_right_y = top_left_y + window_height
        start_point = (top_left_x, top_left_y)
        
        #making heat-map of the full image
        fig, ax = plt.subplots(figsize=(10,10), dpi=100)
        g1 = sns.heatmap(dist_transform, cmap=cmap, cbar=False, yticklabels=False, xticklabels=False)
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        ax.axis('off')
        ax.add_patch(patches.Rectangle(start_point,
                                         window_width,
                                         window_height,
                                         edgecolor="red",
                                         fill = False,
                                         lw=5))
        
        #plt.savefig("two_stage/figures/watershed1.png",dpi=100)
        plt.show()
        
        subwindow = dist_transform[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        #Plotting the subwindow
        
        
        #pixel-plot
        window_size2 = (12, 12)
        window_height2, window_width2 = window_size2
        x_idx2 = 31
        y_idx2 = 19
        top_left_x2 =  x_idx2 - window_width2
        top_left_y2 =  y_idx2 - window_height2
        bottom_right_x2 = top_left_x2 + window_width2
        bottom_right_y2 = top_left_y2 + window_height2
        start_point2 = (top_left_x2, top_left_y2)
        subwindow2 = subwindow[top_left_y2:bottom_right_y2, top_left_x2:bottom_right_x2]
        
        
        fig, ax = plt.subplots(figsize=(10,10))
        
        g1 = sns.heatmap(subwindow, cmap=cmap, cbar=False, yticklabels=False, xticklabels=False)
        g1.set(xlabel=None)
        g1.set(ylabel=None)
        ax.axis('off')
        ax.add_patch(patches.Rectangle(start_point2,
                                         window_width2,
                                         window_height2,
                                         edgecolor="red",
                                         fill = False,
                                         lw=5))
        #plt.savefig("two_stage/figures/watershed2.png", dpi=100)
        plt.show()
        

        
        fig, ax = plt.subplots()
        im = ax.imshow(subwindow2, cmap=cmap)
        fig.colorbar(im)
        for i in range(subwindow2.shape[0]):
            for j in range(subwindow2.shape[1]):
                ax.text(j, i, str(int(subwindow2[i, j])), color='white', ha='center', va='center').set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='white')])
                ax.axis("off")
        #plt.savefig("two_stage/figures/watershed_pixelplot.png", dpi=200)
        plt.show()


        fig, ax = plt.subplots(figsize=(10,10))
        ax.imshow(original_im)
        for cnt in np.unique(markers[markers > 0]):
            temp = markers.copy()
            temp[temp != cnt] = 0
            temp[temp == cnt] = 255
            
            #gray = cv.cvtColor(im,cv.COLOR_RGB2GRAY)
            contours, hierarchy = cv.findContours(np.uint8(temp), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
            new_array = np.squeeze(contours)
            if (len(new_array)) > 2:
                temp=[]
                for j in range(len(new_array)):
                    temp.append(int(new_array[j,0]))
                    temp.append(int(new_array[j,1]))
                ax.plot(temp[0::2],temp[1::2],linestyle="-",linewidth=3)
                
        ax.axis('off')
        #plt.savefig("two_stage/figures/watershed_output.png",dpi=100)
        plt.show()
        
    
    return ret, markers


if __name__ == "__main__":

    r"""
    image_id = 1
    PATH = r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\images"
    image_name = f"window{image_id}.jpg"
    img_name = os.path.join(PATH, image_name)
    
    
    #img_t = preprocess_image(img_name)
    im, img_t = preprocess_image(img_name)
    labels, markers = watershedd(im, img_t)
    
    
    # Loading dataset
    annotation_path = r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\COCO_export.json"
    dataset = load_coco(annotation_path)
    ### Extracting name of the particular grain-type and counting instances in image
    ground_truth = 0
    name = []
    for annotations in dataset["annotations"]:
        if annotations["image_id"]==image_id:
            ground_truth += 1
            for categories in dataset["categories"]:
                if categories["id"] == annotations["category_id"]:
                    #print(annotations["category_id"])
                    name.append(categories["name"])
    name = set(name)
    print("")
    print(f"The following grain-type being analysed is:  {name}   with image_id:  {image_id}")
    print("")
    print(f"The ground-truth amount of kernels in the image is:  {ground_truth}")
    print(f"The amount of kernels the watershed detects is:  {labels}")
    """
    
    image_id = 1
    PATH = r"C:\Users\admin\Desktop\bachelor\Bsc_Thesis_Instance_segmentation\preprocessing\images"
    img_name = r"C:\Users\jver\Desktop\COCO_single\PLS_eval_img_rgb\0_window_Test_Wheat_H1_Dense_series2_20_08_19_13_55_36.jpg"
    #img_name = os.path.join(PATH, image_name)
    
    
    #img_t = binarization(img_name)
    im, img_t = binarization(img_name, plot=True)
    labels, markers = watershedd(im, img_t, 12, plot=True)
    
    
