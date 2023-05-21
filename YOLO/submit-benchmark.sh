#!/bin/sh
#BSUB -J torch_gpu
#BSUB -o benchmark_%J.out
#BSUB -e benchmark_%J.err
#BSUB -n 2
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=16G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10
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

python benchmarks.py --img 256 --data grainSpectral.yaml --weights /work3/coer/Bachelor/yolov5/runs/train-seg/exp150/weights/best.pt
#python segment/train.py --img 256 --batch-size 64 --epochs 300 --data grainSpectral_binary.yaml --weights '' --cfg /work3/coer/Bachelor/yolov5/models/segment/yolov5l-seg.yaml --hyp /work3/coer/Bachelor/yolov5/data/hyps/hyp.no-augmentation.yaml --optimizer "Adam"

#python segment/val.py --weights yolov5s-seg.pt --data grain.yaml --img 640 --name coco


