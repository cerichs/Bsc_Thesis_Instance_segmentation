import os
import numpy as np

PATH = "/work3/coer/Bachelor/datasets/hyperspectral/images"
PATH2 = "/work3/coer/Bachelor/datasets/hyperspectral/corrected/images"
for folders in os.listdir(PATH):
    print(folders)
    for files in os.listdir(PATH+"/"+folders):
        if files.endswith(".npy"):
            
            imgs = np.load(PATH+"/"+folders+"/"+files)
            
            for i in range(102):
                imgs[:,:,i]=((1-0)/(imgs[:,:,i].max()-imgs[:,:,i].min()))*(imgs[:,:,i]-imgs[:,:,i].min())
            #print(imgs.shape)
            assert np.sum(np.isinf(imgs)) == 0
            assert np.sum(np.isnan(imgs)) == 0
            assert np.min(imgs) >=  0
            assert np.max(imgs) <= 1
            #print(np.min(imgs))
            #print(np.max(imgs))
            np.save(PATH2+"/"+folders+"/"+files, imgs)
            
            #print(imgs.shape)
