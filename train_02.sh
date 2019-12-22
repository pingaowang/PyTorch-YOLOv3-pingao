#!/usr/bin/python
#!python

python train.py --model_def config/yolo_cad_dataset_v2.cfg --data_config config/yolo_cad_dataset_v2.data --epochs 2001 --log logs_train_02_init --checkpoint save_train_02_init --lr 0.0001 --img_size 416 --batch_size 6
