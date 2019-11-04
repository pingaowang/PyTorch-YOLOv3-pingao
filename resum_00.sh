#!/usr/bin/python
#!python

python train.py --model_def config/yolov3-custom.cfg --data_config config/custom.data --epochs 201 --log logs_resume_00 --checkpoint save_resume_00 --lr 0.001 --pretrained_weights /save_lr1e-3/yolov3_ckpt_180.pth
