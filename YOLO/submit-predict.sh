#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o exp91_predict_%J.out
#BSUB -e exp91_predict_%J.err
#BSUB -n 1
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=12G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 5
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load matplotlib/3.4.2-numpy-1.21.1-python-3.9.6
module load scipy/1.6.3-python-3.9.6
module load pandas/1.3.1-python-3.9.6
# load CUDA (for GPU support)
module load cuda/11.0

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
source /work3/coer/Bachelor/yolov5/instance-segm-yolo/bin/activate

python segment/predict.py --weights /work3/coer/Bachelor/yolov5/runs/train-seg/exp87/weights/best.pt --img 256 --conf 0.45 --source /work3/coer/Bachelor/datasets/grain256_april/images/test/window1only --hide-label --hide-conf --line-thickness 1

#python segment/val.py --weights yolov5s-seg.pt --data grain.yaml --img 640 --name coco


