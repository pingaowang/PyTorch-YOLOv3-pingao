#!/usr/bin/python
#!python

python train.py --model_def config/yolov3-custom_2.cfg --data_config config/custom.data --epochs 201 --log logs_lr1e-3 --checkpoint save_01 --lr 0.001
