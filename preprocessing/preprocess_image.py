import skimage.util
from skimage.filters import threshold_otsu
import skimage.color
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def binarization(img_name, plot=False):
    
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
        plt.savefig("two_stage/figures/gray_image.png")
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
        plt.savefig("two_stage/figures/histogram.png",dpi=200)
        plt.show()
        
        ## Plotting the binary image
        plt.figure(frameon=False, dpi=100)
        plt.imshow(img_t, cmap="gray")
        plt.axis('off')
        plt.savefig("two_stage/figures/otsu.png")
        plt.show()
    
    return im, img_t
    


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