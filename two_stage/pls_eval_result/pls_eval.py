import sys
sys.path.append("..")

from two_stage.numpy_improved_kernel import PLS
from preprocessing.extract_process_grains import pixel_average, pixel_median
from preprocessing.preprocess_image import binarization
from preprocessing.Display_mask import load_coco
from two_stage.watershed_2_coco import watershed_2_coco
from two_stage.watershed_v2 import watershedd
from two_stage.HSI_mean_msc import mean_centering, msc_hyp, median_centering
from two_stage.pls_watershed import PLS_evaluation, calculate_geometric_mean


from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from PIL import Image
import time
import utils
import torch
import os
import numpy as np
import torch.utils.data
from PIL import Image
import cv2 as cv
import pandas as pd
from skimage.draw import polygon
import transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PLS_eval_img"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PLS_eval_img", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = np.load(img_path)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        # each polygon has unique label
        obj_ids = np.unique(mask)
        # 0 is background
        obj_ids = obj_ids[1:]

        # NCHW, N = # of masks, C = 1, H = W = img.shape
        masks = mask == obj_ids[:, None, None]
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            w = np.max(pos[1])
            ymin = np.min(pos[0])
            h = np.max(pos[0])
            boxes.append([xmin, ymin, w, h])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
        labels_OHE = [int(names in img_path) for names in class_list] # one hot encoding label
        labels_class = labels_OHE.index(1)  # using one hot encoding to get the class number
        labels = torch.ones((num_objs,), dtype=torch.int64)*labels_class # uncomment to evaluate with PLS predictions
        #labels = torch.ones((num_objs,), dtype=torch.int64) #result_dict["labels"].extend([1]) # uncomment to evalaute with watershed (ie. no class predictions)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target, img_path

    def __len__(self):
        return len(self.imgs)

def dataset_prep(dataset,root: str):
    dataset_test = Dataset(root)
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=4, shuffle=False,
        collate_fn=utils.collate_fn) 
    return dataset_test, data_loader_test

def evaluate_model(classifier,dataset, ref):
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_test, data_loader_test = dataset_prep(dataset,r"C:\Users\jver\Desktop\Validation\windows")
    n_threads = torch.get_num_threads()
    cpu_device = torch.device("cpu")
    coco = get_coco_api_from_dataset(dataset_test.dataset)
    iou_types = ["bbox","segm"] # Evaluate both bounding box and segmentation
    coco_evaluator = CocoEvaluator(coco, iou_types)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for images, targets, img_path in metric_logger.log_every(data_loader_test, 100, header):
        images = list(img for img in images)
        model_time = time.time()
        outputs = PLS_class(classifier,images,img_path, dataset, ref) # get output to evalute on
        outputs = [{k: v for k, v in t.items()} for t in outputs] 
        model_time = time.time() - model_time
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)} # add image id for each entry
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

def PLS_class(classifier,images,img_path, dataset, ref):
    color = ["red", "darkblue", "green", "yellow", "white", "orange", "cyan", "pink"]
    class_list = ["Rye_Midsummer", "Wheat_H1", "Wheat_H3",  "Wheat_H4",   "Wheat_H5", "Wheat_Halland",  "Wheat_Oland", "Wheat_Spelt"]
    result = []
    
    
    for image, path in zip(images,img_path):
        spectral_img = image
        fixed_path = path.split("\\")
        fixed_path[-2] = 'PLS_eval_img_rgb' # Uncomment to evaluate with watershed (ie. use predicted masks)
        fixed_path = "\\".join(fixed_path)
        fixed_path = fixed_path[:-3] + 'jpg' # Uncomment to evaluate with watershed (ie. use predicted masks)
        #fixed_path = fixed_path[:-3] + 'png' # Uncomment to evaluate only PLS (ie. use groundtruth masks)
        rgb_image, img_tres = binarization(fixed_path) # Uncomment to evaluate with watershed (ie. use predicted masks)
        
        
        labels, markers = watershedd(rgb_image, img_tres, plot=False) # Uncomment to evaluate with watershed (ie. use predicted masks)
        #markers = Image.open(fixed_path) # Uncomment to evaluate only PLS (ie. use groundtruth masks)
        #markers = np.array(markers) # Uncomment to evaluate only PLS (ie. use groundtruth masks)

        unique_labels = np.unique(markers) # amount of objects in the image
        result_dict = {}
        result_dict["boxes"] = []
        result_dict["labels"] = []
        result_dict["scores"] = []
        result_dict["masks"] = []
        
        for mask_id in np.add(unique_labels, 300)[1:]: # Offset labels to avoid error if mask_id == 255 from Watershed (happens if there are more than 255 grain-kernels)
            mask = markers.copy()
            mask = np.add(mask, 300)
            mask[mask != mask_id] = 0
            mask[mask == mask_id] = 255

            # Compute the pixel average of the spectral image for each grain_mask
            pixel_avg = pixel_average(spectral_img, [mask], None, path.split("\\")[-1])[0]
            #pixel_avg = pixel_median(spectral_img, [mask], None, path.split("\\")[-1])[0]
            
            #Mean-Center
            #mean_center = mean_centering(pd.DataFrame(pixel_avg), ref=ref)
            #pixel_avg = mean_center.values
            
            #Mean-center MSC
            msc = msc_hyp(pd.DataFrame(pixel_avg), ref=ref)[0]
            mean_msc = mean_centering(msc, ref=ref)
            pixel_avg = mean_msc.values
            
            #Median-Center
            #median_center = median_centering(pd.DataFrame(pixel_avg), ref=ref)[0]
            #pixel_avg = median_center.values
            

            # Get prediction for the mask
            result2 = classifier.predict(pixel_avg, A=5)

            contours, _ = cv.findContours(np.uint8(mask), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) # CHAIN_APPROX_NONE to avoid RLE
            if len(contours) != 1 or (len(np.squeeze(contours))) <= 2:
                print(fixed_path)
            if (len(contours) == 1) and (len(np.squeeze(contours))) > 2:
                anno = watershed_2_coco(contours)
                #plt.fill(anno[0::2],anno[1::2], color=color[np.argmax(result2)])
                x = min(anno[0::2])
                y = min(anno[1::2])
                w = max(anno[0::2])
                h = max(anno[1::2])
                result_dict["boxes"].append([x,y,w,h]) # COCO functions changes to width and height, therefore not done here
                result_dict["labels"].extend([np.argmax(result2)]) # Uncomment to evaluate with PLS (ie. use predicted classes)
                #result_dict["labels"].extend([1.0]) # uncomment to see ONLY watershed with no PLS (ie. no predicted classes)
                result_dict["scores"].extend([1.0]) # set higher than conf threshold
                result_dict["masks"].append(np.expand_dims(mask/255, axis=-3)) # HW to CHW

        result_dict["boxes"] = torch.tensor(result_dict["boxes"])
        result_dict["labels"] = torch.tensor(result_dict["labels"])
        result_dict["scores"] = torch.tensor(result_dict["scores"])
        result_dict["masks"] = torch.tensor(result_dict["masks"])
        result.append(result_dict)
    return result
    
def coco_2_masked_img(dataset_path):
    dataset = load_coco(dataset_path)
    n = len(dataset["images"])
    for i in range(n):
        empty_mask = np.zeros((256,256))
        label = 1 # 0 is background
        for j, anno in enumerate(dataset["annotations"]):
            if anno["image_id"] == i:
                x, y = (anno["segmentation"][0][0::2]),(anno["segmentation"][0][1::2])
                row, col = polygon(y, x)
                empty_mask[row,col] = label
                label += 1
        cv.imwrite(f"C:/Users/jver/Desktop/Validation/windows/masks/{dataset['images'][i]['file_name'][:-4]}.png",empty_mask)
        #plt.imshow(empty_mask,interpolation="none")
        #plt.savefig(f"I:/HSI/masks/{dataset['images'][i]['file_name'][:-4]}.png")



def main():
    coco_2_masked_img(r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json") # uncomment to make PNG images with masks from a COCO json file
    #data = pd.read_csv(r"C:\Users\Corne\Downloads\Pixel_grain_avg_dataframe_train_whole_img.csv")
    data = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\Pixel_grain_avg_dataframe_train_meanMSC_whole_img.csv")
    ref_average = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\MSC.csv", header=None)
    ref_average = ref_average.iloc[:,1].values[1:]
    ref_median = pd.read_csv(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\median.csv", header=None)
    ref_median = ref_median.iloc[:,1].values[1:]
    X = data.iloc[:,2:]
    y_test = data.label
    Y = [eval(y_test[i]) for i in range(len(y_test))]
    classifier = PLS(algorithm=2)
    classifier.fit(X, Y, 102)
    #classifier.predict(X, A=20)
    dataset = Dataset(r'C:\Users\jver\Desktop\Validation\windows')
    evaluate_model(classifier,dataset, ref_average)


if __name__ == "__main__":
    main()