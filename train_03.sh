#!/usr/bin/python
#!python

python train.py --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 2001 --log logs_train_03_init --checkpoint save_train_03_init --lr 0.0001 --img_size 416 --batch_size 6
