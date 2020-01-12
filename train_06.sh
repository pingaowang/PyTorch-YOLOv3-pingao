#!/usr/bin/python
#!python

python train.py --aug_crop 0.1 --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 1001 --log logs_train_06_init --checkpoint save_train_06_init --lr 0.0001 --img_size 416 --batch_size 24 --checkpoint_interval 200
python train.py --aug_crop 0.1 --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 1001 --log logs_train_06_r1 --checkpoint save_train_06_r1 --lr 0.00001 --img_size 416 --batch_size 24 --checkpoint_interval 200  --pretrained_weights save_train_06_init/yolov3_ckpt_1000.pth
python train.py --aug_crop 0.1 --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 1001 --log logs_train_06_r2 --checkpoint save_train_06_r2 --lr 0.0001 --img_size 416 --batch_size 24 --checkpoint_interval 200  --pretrained_weights save_train_06_r1/yolov3_ckpt_1000.pth
python train.py --aug_crop 0.1 --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 1001 --log logs_train_06_r3 --checkpoint save_train_06_r3 --lr 0.00001 --img_size 416 --batch_size 24 --checkpoint_interval 200  --pretrained_weights save_train_06_r2/yolov3_ckpt_1000.pth
python train.py --aug_crop 0.1 --model_def config/yolo_cad_dataset_v3.cfg --data_config config/yolo_cad_dataset_v3.data --epochs 1001 --log logs_train_06_r4 --checkpoint save_train_06_r4 --lr 0.000001 --img_size 416 --batch_size 24 --checkpoint_interval 200  --pretrained_weights save_train_06_r3/yolov3_ckpt_1000.pth
