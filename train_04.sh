#!/usr/bin/python
#!python

python train.py --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v4.data --epochs 2001 --log logs_train_04_init --checkpoint save_train_04_init --lr 0.0001 --img_size 416 --batch_size 24 --checkpoint_interval 10
