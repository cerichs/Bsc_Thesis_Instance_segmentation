# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:54:38 2023

@author: Cornelius
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# im sorry for this, please forgive me

def add_text_to_predict(img): 
    colors = [(56, 56, 255), (151, 157, 255) , (31, 112, 255), (29, 178, 255), (49, 210, 207),(10, 249, 72),(23, 204, 146),(134, 219, 61)] # BGR format
    #class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    img = cv.imread(img)
    #img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    height, width, channels = img.shape

    N = len(class_list)
    color_text_img = np.zeros((height, 260, 3), dtype=np.uint8)
    for i, (color, text) in enumerate(zip(colors, class_list)):

        y = i * (height // N)
        
        cv.rectangle(color_text_img, (0, y), (260, y + height // N), color, -1)

        cv.putText(color_text_img, text, (5, y - 5 + height // (N)), cv.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)

    new_img = np.hstack((img, color_text_img))

    cv.imwrite('new_image_test.jpg', new_img)

def add_legend_to_predict(img):
    colors = [(255, 56, 56), (255, 157, 151) , (255, 112, 31), (255, 178, 29), (207, 210, 49),(72, 249, 10),(146, 204, 23),(61, 219, 134)] # RGB Format
    colors = [tuple(np.array(color)/255) for color in colors]
    #class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    img = cv.imread(img)
    plt.figure(dpi=200)
    plt.imshow(img)

    handles = [plt.Rectangle((0,0),1,1, color=color, ec="k") for color in colors]  # Create the colored rectangles for the legend
    plt.legend(handles, class_list, loc="center left", bbox_to_anchor =(1,0.5), framealpha=1)  # Add the legend to the plot
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    img_path = r"C:\Users\Cornelius\Documents\GitHub\Bscproject\Bsc_Thesis_Instance_segmentation\window0.jpg"
    add_text_to_predict(img_path)
    add_legend_to_predict(img_path)