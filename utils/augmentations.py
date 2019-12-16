import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2


def show(tensor, mode='RGB'):
    if type(tensor) == np.ndarray:
        transforms.ToPILImage(mode=mode)(torch.from_numpy(tensor)).show()
    elif type(tensor) == torch.Tensor:
        transforms.ToPILImage(mode=mode)(tensor).show()


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def test(images, targets):
    ## images
    arr_img = np.array(images)
    # show(arr_img)

    arr_targets = np.array(targets)
    c, h, w = arr_img.shape
    scale = [2., 0.5]

    # resize: (3, 400, 400)
    resize_h = int(h * scale[0])
    resize_w = int(w * scale[0])
    img_resize = np.ones((3, resize_h, resize_w))
    img_resize[:, int((resize_h - h) / 2): int((resize_h - h) / 2 + h), int((resize_w - w) / 2): int((resize_w - w) / 2 + w)] = arr_img
    img_resize  = img_resize.astype(np.float32)

    # config of crop
    max_crop_size = (int(scale[0] * h), int(scale[0] * w))
    min_crop_size = (int(scale[1] * h), int(scale[1] * w))
    crop_h = np.random.randint(max_crop_size[0] - min_crop_size[0]) + min_crop_size[0]
    # crop_w = np.random.randint(max_crop_size[1] - min_crop_size[1]) + min_crop_size[1]
    crop_w = crop_h

    top_left_loc = (
        np.random.randint(resize_h - crop_h),
        np.random.randint(resize_w - crop_w)
        )

    # crop
    img_crop = img_resize[:, top_left_loc[0]: top_left_loc[0] + crop_h, top_left_loc[1]: top_left_loc[1] + crop_w]

    # resize back: (3, 200, 200)
    arr_img_resize_back = cv2.resize(img_crop.transpose((1, 2, 0)), (h, w)).transpose((2, 0, 1))
    img_resize_back = torch.from_numpy(arr_img_resize_back)
    # show(img_resize_back)

    ## targets
    arr_targets = np.array(targets)
    list_targets_new = list()
    n_targets = arr_targets.shape[0]

    for i in range(n_targets):
        target = arr_targets[i, :]
        x_1 = (target[2] * w)
        x_2 = x_1 + (resize_w - w) / 2
        x_3 = (x_2 - top_left_loc[1]) / crop_w

        y_1 = target[3] * h
        y_2 = y_1 + (resize_h - h) / 2
        y_3 = (y_2 - top_left_loc[0]) / crop_h

        # print("({}, {})".format(x_3, y_3))

        w_1 = target[4]
        h_1 = target[5]
        bb_w = w_1 * w / crop_w
        bb_h = h_1 * h / crop_h
        target_new = np.array([
            target[0],
            target[1],
            x_3,
            y_3,
            bb_w,
            bb_h
        ])
        if 0 < x_3 < 1 and 0 < y_3 < 1:
            list_targets_new.append(target_new)

    if len(list_targets_new) != 0:
        arr_targets_new = np.array(list_targets_new).astype(np.float32)
        targets_new = torch.from_numpy(arr_targets_new)
        return img_resize_back, targets_new
    else:
        return images, targets


## Crop

## Color






