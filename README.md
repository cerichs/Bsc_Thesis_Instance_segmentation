# Bachelor Thesis, Instance segmentation with FOSS

This repository contains the code used in the bachelor thesis. <br>
Made by:<br>
Cornelius Erichs, Artificial Intelligence and Data (B.Sc.), Technical University of Denmark (DTU)<br>
Johan Verrecchia, Artificial Intelligence and Data (B.Sc.), Technical University of Denmark (DTU)<br>

## <div align="center">Dataset download</div>
The Pseudo-RGB dataset can be downloaded [HERE](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/releases/download/Release/PseudoRGB.zip) the HSI dataset can be downloaded [HERE](https://sid.erda.dk/share_redirect/e4Wr5LFI4L). Due to the size limitations on GitHub, we are unable to upload the version of the HSI dataset that only contains the annotated images, the full dataset will therefore have to be downloaded.

## <div align="center">Running Two-Stage Approach</div>
The two-stage approach aswell as the data generation using cropped subwindows and synthetic data, can be run through [main.py](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/main.py). 

**Important**, the Two-Stage approach uses a non published PLS-DA algorithm implementation by Ole-Christian Galbo Engstrøm from FOSS (ocge@foss.dk). It is therefore not in this repository, the script can be accessed in the Teams channel used for meetings and contact.

## <div align="center">Running YOLO</div>
The YOLOv5 Pseudo-RGB training ([submit.sh](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/submit.sh)), validation ([submit-val.sh](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/submit-val.sh)) and prediction ([submit-predict.sh](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/submit-predict.sh)) scripts have to be run through the commandline.

**Important**, due to both Pseudo-RGB and HSI training scripts are contained within this, 5 files have to be renamed to train Pseudo-RGB or HSI.

To train, validate and predict using Pseudo-RGB change the following:
```command
YOLO/segment/train-RGB.py -> YOLO/segment/train.py
YOLO/segment/val-RGB.py -> YOLO/segment/val.py
YOLO/segment/predict-RGB.py -> YOLO/segment/predict.py
YOLO/utils/dataloaders-RGB.py -> YOLO/utils/dataloaders.py
YOLO/utils/segment/dataloaders-RGB.py -> YOLO/utils/segment/dataloaders.py
```
To train, validate and predict using HSI change the following:

```command
YOLO/segment/train-HSI.py -> YOLO/segment/train.py
YOLO/segment/val-HSI.py -> YOLO/segment/val.py
YOLO/segment/predict-HSI.py -> YOLO/segment/predict.py
YOLO/utils/dataloaders-HSI.py -> YOLO/utils/dataloaders.py
YOLO/utils/segment/dataloaders-HSI.py -> YOLO/utils/segment/dataloaders.py
```

The scripts to run Pseudo-RGB don't have any changes compared to the official YOLOv5 repository.
The changes made to run on HSI are in the 5 files mentioned above, all the rest is untouched.

To run YOLO either follow the documentation from Ultralytics in [README.md](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/README.md) or use the command line below:

(Assuming the terminal current directory is the YOLO folder, requirements.txt have been installed in your virtual environment)

```command
python segment/train.py --img 256 --batch-size 32 --epochs 400 --data grain256_april.yaml --weights '' --cfg /work3/coer/Bachelor/yolov5/models/segment/yolov5xn-seg.yaml --cache
```

```command
python segment/val.py --weights /PATH/TO/BEST/WEIGHT/best.pt --data grainSpectral.yaml --img 256 --task "test"
```

```command
python segment/predict.py --weights /PATH/TO/BEST/WEIGHT/best.pt --img 256 --conf 0.45 --source /PATH/TO/FOLDER/WITH/IMAGES/TO/PREDICT --hide-label --hide-conf --line-thickness 1
```

Running the HSI training will give errors as it is unable to plot HSI data, this does not affect the training and can be ignored.

## <div align="center">Convert COCO to YOLO</div>
The convert a .json file in COCO format, we recommend using the script by Ultralytics, which has been modified to fit our classes. Can be accesed [Here](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/optimize/YOLO/JSON2YOLO-master/general_json2yolo.py). 

Currently it converts the COCO json file to Classification to convert to Binary classification the following [codesnippet](https://github.com/cerichs/Bsc_Thesis_Instance_segmentation/blob/99147cf4bc32efdf3554be6aaebd2a6cac800488/YOLO/JSON2YOLO-master/general_json2yolo.py#LL48C17-L52C43) has to be changed (line 48-52)


```python
class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
#            [Rye_midsummer, Wheat_H1, Wheat_H3,  Wheat_H4,   Wheat_H5, Wheat_Halland,  Wheat_Oland, Wheat_Spelt, Foreign]
cls = class_list.index(cls)
#cls = 0
box = [cls] + box.tolist()
```
To make the binary classification labels change it to:

```python
class_list = [ 1412692,     1412693,   1412694,   1412695,    1412696,     1412697,      1412698,    1412699,     1412700]
#            [Rye_midsummer, Wheat_H1, Wheat_H3,  Wheat_H4,   Wheat_H5, Wheat_Halland,  Wheat_Oland, Wheat_Spelt, Foreign]
#cls = class_list.index(cls)
cls = 0
box = [cls] + box.tolist()
```