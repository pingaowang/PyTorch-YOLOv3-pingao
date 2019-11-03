#!/usr/bin/python
#!python

python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --log logs_lr1 --lr 1.0
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --log logs_lr1e-1 --lr 0.1
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --log logs_lr1e-2 --lr 0.01
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --log logs_lr1e-3 --lr 0.001
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --log logs_lr1e-4 --lr 0.0001
