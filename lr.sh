#!/usr/bin/python
#!python

python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 200 --log logs_lr1e-3 --checkpoint save_lr1e-3 --lr 0.001
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 200 --log logs_lr1e-4 save_lr1e-4 --lr 0.0001
python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 200 --log logs_lr1e-5 save_lr1e-5 --lr 0.00001
