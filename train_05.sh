#!/usr/bin/python
#!python

#python train.py --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 2001 --log logs_train_05_init --checkpoint save_train_05_init --lr 0.0001 --img_size 416 --batch_size 24 --checkpoint_interval 50
python train.py --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 20001 --log logs_train_05_r1 --checkpoint save_train_04_r1 --lr 0.00001 --img_size 416 --batch_size 24 --checkpoint_interval 500  --pretrained_weights save_train_05_init/yolov3_ckpt_2000.pth
