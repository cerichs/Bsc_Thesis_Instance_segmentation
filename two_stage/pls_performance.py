from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
import numpy as np
import json

#path = r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\PLS_coco_Test Original whole_img.json"
path = r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\PLS_coco_Test Original grain.json"


with open(path) as f:
    dataset = json.load(f)
    f.close()
#coco_temp = COCO(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\PLS_coco_Test Original whole_img.json")
coco_temp = COCO(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\PLS_coco_Test Original grain.json")


results = []
annids = coco_temp.getAnnIds()
anns = coco_temp.loadAnns(annids)

for i, ann in enumerate(dataset["annotations"]):
    ann["image_id"] += 1
    anns[i]["image_id"] += 1
    mask_encode = mask.encode(np.asfortranarray(coco_temp.annToMask(anns[i])))

    anndata = {}
    anndata['image_id'] = ann["image_id"]
    anndata['category_id'] = ann["category_id"]
    anndata['segmentation'] = mask_encode
    anndata['area'] = ann["area"]
    anndata['score'] = 1.0
    results.append(anndata)



#true_coco = COCO(r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json")
true_coco = COCO(r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json")
pred_coco = true_coco.loadRes(results)
imgIds = sorted(true_coco.getImgIds())
cocoEval = COCOeval(true_coco, pred_coco, 'segm')
#cocoEval.params.catIds = [1412700] #person id : 1
cocoEval.params.imgIds = imgIds
cocoEval.params.maxDets = [800]* len(cocoEval.params.iouThrs)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()






ground_truth_file = r"C:\Users\jver\Desktop\Validation\windows\COCO_HSI_windowed.json"
predictions_file = r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation\two_stage\pls_results\PLS_coco_Test Original grain.json"

# Load the ground truth and predictions JSON files
with open(ground_truth_file, 'r') as gt_file, open(predictions_file, 'r') as pred_file:
    gt_data = json.load(gt_file)
    pred_data = json.load(pred_file)

# Initialize COCO ground truth API
gt_coco = COCO(ground_truth_file)

# Load the 'annotations' part from the ground truth JSON
gt_annotations = gt_data['annotations']

# Load the 'annotations' part from the predictions JSON
pred_annotations = pred_data['annotations']

# Function to convert a list of RLE objects to a list of dictionaries
def rle_to_dict_list(rle_list):
    rle_dict_list = []
    for rle in rle_list:
        rle_dict = {
            'counts': rle['counts'].decode('ascii'),
            'size': rle['size']
        }
        rle_dict_list.append(rle_dict)
    return rle_dict_list

valid_gt_annotations = []
valid_pred_annotations = []

for gt_ann, pred_ann in zip(gt_annotations, pred_annotations):
    gt_polygon = gt_ann['segmentation']
    pred_polygon = pred_ann['segmentation']

    try:
        gt_rle_list = mask.frPyObjects(gt_polygon, 255, 255)
        pred_rle_list = mask.frPyObjects(pred_polygon, 255, 255)

        gt_binary_mask = mask.decode(gt_rle_list)
        pred_binary_mask = mask.decode(pred_rle_list)

        gt_rle_list = mask.encode(gt_binary_mask)
        pred_rle_list = mask.encode(pred_binary_mask)

        gt_ann['segmentation'] = rle_to_dict_list(gt_rle_list)
        pred_ann['segmentation'] = rle_to_dict_list(pred_rle_list)

        valid_gt_annotations.append(gt_ann)
        valid_pred_annotations.append(pred_ann)
    except Exception as e:
        print(f"Error encountered for annotation: {gt_ann}")
        print(f"Error message: {str(e)}")

# Save the updated annotations to new JSON files
with open('ground_truth_rle.json', 'w') as gt_output, open('predictions_rle.json', 'w') as pred_output:
    json.dump(valid_gt_annotations, gt_output)
    json.dump(valid_pred_annotations, pred_output)

# Load the COCO ground truth and predictions
gt_coco = COCO('ground_truth_rle.json')
pred_coco = gt_coco.loadRes('predictions_rle.json')

# Create COCO evaluation object
coco_eval = COCOeval(gt_coco, pred_coco, iouType='segm')

# Evaluate and accumulate the results
coco_eval.evaluate()
coco_eval.accumulate()

# Compute and display the mAP score
coco_eval.summarize()

