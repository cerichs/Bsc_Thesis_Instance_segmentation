from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask
import numpy as np
import json

path = r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation-main\preprocessing\pls_results\PLS_coco_Test Original grain.json"



with open(path) as f:
    dataset = json.load(f)
    f.close()
coco_temp = COCO(r"C:\Users\jver\Desktop\dtu\Bsc_Thesis_Instance_segmentation-main\Bsc_Thesis_Instance_segmentation-main\preprocessing\pls_results\PLS_coco_Test Original grain.json")

results = []
annids = coco_temp.getAnnIds()
anns = coco_temp.loadAnns(annids)

for i, ann in enumerate(dataset["annotations"]):	
	mask_encode = mask.encode(np.asfortranarray(coco_temp.annToMask(anns[i])))
	anndata = {}
	anndata['image_id'] = ann["image_id"]
	anndata['category_id'] = ann["category_id"]
	anndata['segmentation'] = mask_encode
	anndata['area'] = ann["area"]
	anndata['score'] = 1.0
	results.append(anndata)



true_coco = COCO(r"C:\Users\jver\Desktop\dtu\DreierHSI_Apr_05_2023_10_11_Ole-Christian Galbo\Validation\COCO_Validation.json")

pred_coco = true_coco.loadRes(results)
imgIds = sorted(true_coco.getImgIds())
cocoEval = COCOeval(true_coco, pred_coco, 'segm')
#cocoEval.params.catIds = [1412700] #person id : 1
cocoEval.params.imgIds = imgIds
cocoEval.params.maxDets = [800]* len(cocoEval.params.iouThrs)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()