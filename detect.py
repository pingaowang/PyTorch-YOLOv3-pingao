from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import cv2

from utils.draw_bb import draw_bb

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/mini/images/", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-mini.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/mini/classes.names", help="path to class label file")
    parser.add_argument("--color_path", type=str, default="data/mini/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.9, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.2, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)


    """"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file
    list_color = load_color(opt.color_path)
    assert len(classes) == len(list_color), "The number of classes and colors are not the same. please check {} and {}".format(opt.class_path, opt.color_path)
    print(list_color)

    """"""
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # Tensor = torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

            if detections[0] is None:
                print("There are No blocks.")
            else:
                print("There are {} blocks.".format(detections[0].size()[0]))

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    dict_img_out = {}
    l_filename = []
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))
        img = cv2.imread(path)
        img_size = img.shape[0]

        filename = path.split("/")[-1].split(".")[0]
        l_filename.append(filename)

        # Create plot
        # img = np.array(Image.open(path))
        # plt.figure()
        # fig, ax = plt.subplots(figsize=(20, 10))
        # ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            # bbox_colors = random.sample(colors, n_cls_preds)
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        #
        #         print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
        #
        #         box_w = x2 - x1
        #         box_h = y2 - y1
        #
        #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        #         # Create a Rectangle patch
        #         bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=3, edgecolor=color, facecolor="none")
        #         # Add the bbox to the plot
        #         ax.add_patch(bbox)
        #         # Add label
        #         plt.text(
        #             x1,
        #             y1,
        #             s=classes[int(cls_pred)],
        #             color="white",
        #             verticalalignment="top",
        #             bbox={"color": color, "pad": 0},
        #         )
        #
        # # Save generated image with detections
        # plt.axis("off")
        # plt.gca().xaxis.set_major_locator(NullLocator())
        # plt.gca().yaxis.set_major_locator(NullLocator())
        # filename = path.split("/")[-1].split(".")[0]
        # plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        # plt.close()
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                txt = "{0}: conf {1:.2f}, {2:.2f}".format(classes[int(cls_pred)], conf, cls_conf)
                color = list_color[int(cls_pred)]

                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                img_out = draw_bb(img=img, x1=x1, y1=y1, x2=x2, y2=y2, txt=' ', color=color, thickness=2)
        else:
            img_out = img


        dict_img_out.update(
            {filename: img_out}
        )

        # cv2.imwrite(f"output/{filename}.png", img_out)


    ## combine blocks to a whole image.
    l_in_block_row = list()
    for filename in l_filename:
        l_in_block_row.append(int(filename.split('_')[0]))
    n_blocks_row = np.max(l_in_block_row) + 1
    img_whole_size = n_blocks_row * img_size
    img_whole = np.zeros((img_whole_size, img_whole_size, 3))

    for i in range(n_blocks_row):
        for j in range(n_blocks_row):
            filename = "{}_{}".format(i, j)
            img_whole[i * img_size: (i+1) * img_size, j * img_size: (j+1) * img_size] = dict_img_out[filename]

    cv2.imwrite(f"output/whole.png", img_whole)








