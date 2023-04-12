import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
import os
#from watershed_2_coco import coco_dict, export_json
from Display_mask import load_coco, load_annotation, find_image, draw_img
import skimage.color
import skimage.util
import seaborn as sns
import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import matplotlib.patches as patches

    
def preprocess_image(img_name, plot=False):
    
    im = cv.imread(img_name) # Load image

    im = cv.cvtColor(im,cv.COLOR_BGR2RGB) # Load image
    grayscale = cv.cvtColor(im,cv.COLOR_RGB2GRAY) # Convert to Grayscale


    #Converting image to float - depicting gray-scale intensities
    image_g = skimage.util.img_as_float(grayscale)
    # image histogram - used in explaining Otsu's method
    histogram, bin_edges = np.histogram(image_g, bins=256)

    #Otsu's method for thresholding
    auto_tresh = threshold_otsu(image_g) # Determine Otsu threshold
    #Plotting histogram, threshold value and 

    segm_otsu = (image_g > auto_tresh) # Apply threshold
    img_t = segm_otsu.astype(int)*255 # Convert Bool type to 0:255 int
    
    
    if plot:
        
        ## Original image
        plt.figure(frameon=False, dpi=100)
        plt.imshow(grayscale, cmap="gray")
        plt.axis('off')
        plt.savefig("image.png")
        plt.show()
        
        
        ## Grayscale image
        plt.figure(frameon=False, dpi=100)
        plt.imshow(grayscale, cmap="gray")
        plt.axis('off')
        plt.savefig("gray_image.png")
        plt.show()
        
        ## Grayscale histogram and Otsu threshold
        plt.figure(frameon=False, dpi=100)
        plt.title("Grayscale Histogram")
        plt.xlabel("grayscale value")
        plt.ylabel("pixel count")
        plt.xlim()
        #plotting the y-line of the histogram
        plt.axvline(auto_tresh, 0, 1, label="Otsu's Threshold", c="Red")
        plt.legend(["Otsu's Threshold"])
        #plotting the histogram
        sns.histplot(image_g.ravel(), binrange=(0,1), bins=100)
        plt.savefig("histogram.png",dpi=200)
        plt.show()
        
        ## Plotting the binary image
        plt.figure(frameon=False, dpi=100)
        plt.imshow(img_t, cmap="gray")
        plt.axis('off')
        plt.savefig("otsu.png")
        plt.show()
    
    return im, img_t
    

def watershedd(original_im, image_t, plot=False):
    # Finding foreground area
    # Finds the smallest distance between object and background
    dist_transform = cv.distanceTransform(np.uint8(image_t),cv.DIST_L2,5)
    
    l_max = peak_local_max(dist_transform, indices=False, min_distance=11,labels=image_t)
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
        cvals = [-i for i in range(int(np.max(dist_transform)))][::-1][::2]
        colors = ["midnightblue","navy", "darkblue", "mediumblue", "royalblue", "deepskyblue", "cyan", "black"][::-1]
        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("", tuples)
        
        #creating subwindow of image (pretty centered - but one can change x_idx and y_idx)
        window_size = (32, 32)
        window_height, window_width = window_size
        x_idx = 250
        y_idx = 200
        top_left_x =  x_idx - window_width
        top_left_y =  y_idx - window_height
        bottom_right_x = top_left_x + window_width
        bottom_right_y = top_left_y + window_height
        start_point = (top_left_x, top_left_y)
        
        #making heat-map of the full image
        #fig, ax = plt.subplots(figsize=(10,10), dpi=100)
        #g1 = sns.heatmap(dist_transform, cmap=cmap, cbar=False, yticklabels=False, xticklabels=False)
        #g1.set(xlabel=None)
        #g1.set(ylabel=None)
        #ax.axis('off')
        #ax.add_patch(patches.Rectangle(start_point,
        #                                 window_width,
        #                                 window_height,
        #                                 edgecolor="red",
        #                                 fill = False,
        #                                 lw=5))
        #
        #plt.savefig("watershed1.png",dpi=100)
        #plt.show()
        
        """
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
        size_ratio = window_size[0]/len(image_t)
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
        plt.savefig("watershed2.png", dpi=100)
        plt.show()
        

        
        fig, ax = plt.subplots()
        im = ax.imshow(subwindow2, cmap=cmap)
        fig.colorbar(im)
        for i in range(subwindow2.shape[0]):
            for j in range(subwindow2.shape[1]):
                ax.text(j, i, str(int(subwindow2[i, j])), color='white', ha='center', va='center').set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='white')])
                ax.axis("off")
        plt.savefig("watershed_pixelplot.png", dpi=200)
        plt.show()
        """
        """
        plt.title("pixel_plot")
        pixel_plot = plt.imshow(
          subwindow, cmap=cmap, interpolation='nearest', origin='lower')
        plt.colorbar(pixel_plot)
          
        plt.savefig("watershed_pixelplot.png", dpi=400)
        plt.show(pixel_plot)
        ax.axis('off')
        """
        
        plt.imshow(markers)
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
        plt.savefig("watershed_output.png",dpi=100)
        plt.show()
        
    
    return ret, markers


if __name__ == "__main__":
    
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
    
        



#test_dict = coco_dict(img_name,markers)

#export_json(test_dict)
#plt.imshow(im)
#plt.fill(x,y,alpha=.7,color='g')
#plt.show()

# =============================================================================
# fix,(ax1,ax2)=plt.subplots(1,2)
# im[markers == -1] = [255,0,0]
# ax1.imshow(im)
# ax1.axis('off')
# ax2.imshow(markers == -1)
# ax2.axis('off')
# plt.show()
# =============================================================================
