#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o export_run115_%J.out
#BSUB -e export_run115_%J.err
#BSUB -n 2
#BSUB -R "rusage[mem=32G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load matplotlib/3.4.2-numpy-1.21.1-python-3.9.6
module load scipy/1.6.3-python-3.9.6
module load pandas/1.3.1-python-3.9.6
# load CUDA (for GPU support)
module load cuda/11.7

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source /work3/coer/Bachelor/yolov5/instance-segm-yolo/bin/activate

python export.py --img 256 --data grainSpectral.yaml --weights /work3/coer/Bachelor/yolov5/runs/train-seg/exp115/weights/best.pt --include onnx


#python segment/val.py --weights yolov5s-seg.pt --data grain.yaml --img 640 --name coco





