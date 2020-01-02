import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import utils.augmentations as Aug
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    # def __init__(self, list_path, img_size=416, augment=True, multiscale=True, normalized_labels=True):
    def __init__(
            self, list_path, img_size=416, multiscale=True, normalized_labels=True,
            aug_black_white=True,
            aug_rotate=(0, 360),
            # aug_rotate=False,
            aug_gaussian_noise=0.2,
            aug_random_hv_flip=True,
            aug_normalize=False,
            aug_random_resize_crop=(0.7, 1.5),
            aug_color_jitter={'b': 0.2, 'c': 0.2, 's': 0.2, 'h': 0.3}
    ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        # self.augment = augment
        self.aug_black_white = aug_black_white
        self.aug_rotate = aug_rotate
        self.aug_gaussian_noise = aug_gaussian_noise
        self.aug_random_hv_flip = aug_random_hv_flip
        self.aug_normalize = aug_normalize
        self.aug_random_resize_crop = aug_random_resize_crop
        self.aug_color_jitter = aug_color_jitter

        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = Image.open(img_path).convert('RGB')
        tensor_img = transforms.ToTensor()(np.array(img))

        # Handle images with less than three channels
        if len(tensor_img.shape) != 3:
            tensor_img = tensor_img.unsqueeze(0)
            tensor_img = tensor_img.expand((3, tensor_img.shape[1:]))

        _, h, w = tensor_img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        tensor_img, pad = pad_to_square(tensor_img, 0)
        _, padded_h, padded_w = tensor_img.shape


        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            # print(w_factor)
            # print(padded_w)
            # print(boxes[:, 3])
            # boxes[:, 3] *= w_factor / padded_w
            # print(boxes[:, 3])
            # boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes

        # print(targets)
        ## Data augmentations
        # if self.augment:
        # if np.random.random() < 0.5:
        # img, targets = horisontal_flip(img, targets)
        # img, targets = test(img, targets)

        # aug_random_resize_crop
        img, targets = Aug.test(img, targets, scale_range=self.aug_random_resize_crop)

        # aug_rotate
        # img.show()
        # print(targets)
        if self.aug_rotate:
            img, targets = Aug.random_rotate(img, targets, self.aug_rotate)

        # aug_random_hv_flip

        # aug_color_jitter
        if self.aug_color_jitter:
            img = Aug.color_jitter(img, self.aug_color_jitter)

        # aug_gaussian_noise

        # aug_black_white
        if self.aug_black_white:
            img = Aug.black_white(img)

        img.show()
        # print(targets)

        img = transforms.ToTensor()(img)

        img = resize(img, self.img_size)
        img = img - 0.5

        # print("targets: {}".format(targets))
        targets = targets.float()
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            # print(boxes)
            # print(boxes[:, 0])
            boxes[:, 0] = i

        # if len(targets) == 0:
        #     targets = [torch.zeros((1, 6), dtype=torch.float)]
        targets = torch.cat(targets, 0)

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)
