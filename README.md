# Bachelor Thesis, Instance segmentation with FOSS

This repository contains the code used in the bachelor thesis. The two-stage approach aswell as the data generation using cropped subwindows and synthetic data, can be run through [main.py](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/main.py). 

## <div align="center">Dataset download</div>
The Pseudo-RGB dataset can be downloaded [HERE](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/releases/download/Release/PseudoRGB.zip) the HSI dataset can be downloaded [HERE](https://sid.erda.dk/share_redirect/e4Wr5LFI4L). Due to the size limitations on GitHub, we are unable to upload the version of the HSI dataset that only contains the annotated images, the full dataset will therefore have to be downloaded.

## <div align="center">Running YOLO</div>
The YOLOv5 Pseudo-RGB training ([submit.sh](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/submit.sh)), validation ([submit-val.sh](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/submit-val.sh)) and prediction ([submit-predict.sh](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/submit-predict.sh)) scripts have to be run through the commandline.

IMPORTANT, due to both Pseudo-RGB and HSI training scripts are contained within this, 5 files have to be renamed to train Pseudo-RGB or HSI.

To train, validate and predict using Pseudo-RGB change the following:
```command
YOLO/segment/train-RGB.py -> YOLO/segment/train.py
YOLO/segment/val-RGB.py -> YOLO/segment/val.py
YOLO/segment/predict-RGB.py -> YOLO/segment/predict.py
YOLO/utils/dataloaders-RGB.py -> YOLO/utils/dataloaders.py
YOLO/utils/segment/dataloaders-RGB.py -> YOLO/utils/dataloaders.py
```
To train, validate and predict using HSI change the following:

```command
YOLO/segment/train-HSI.py -> YOLO/segment/train.py
YOLO/segment/val-HSI.py -> YOLO/segment/val.py
YOLO/segment/predict-HSI.py -> YOLO/segment/predict.py
YOLO/utils/dataloaders-HSI.py -> YOLO/utils/dataloaders.py
YOLO/utils/segment/dataloaders-HSI.py -> YOLO/utils/dataloaders.py
```
To run YOLO either follow the documentation from Ultralytics in [README.md](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/README.md) or use the command line below:

(Assuming the terminal current directory is the YOLO folder)

```command
python segment/train.py --img 256 --batch-size 32 --epochs 400 --data grain256_april.yaml --weights '' --cfg /work3/coer/Bachelor/yolov5/models/segment/yolov5xn-seg.yaml --cache
```

```command
python segment/val.py --weights /PATH/TO/BEST/WEIGHT/best.pt --data grainSpectral.yaml --img 256 --task "test"
```

```command
python segment/predict.py --weights /PATH/TO/BEST/WEIGHT/best.pt --img 256 --conf 0.45 --source /PATH/TO/FOLDER/WITH/IMAGES/TO/PREDICT --hide-label --hide-conf --line-thickness 1
```