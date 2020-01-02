import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageDraw
import cv2
from utils.rotate import rotate_angle


def show(tensor, mode='RGB'):
    if type(tensor) == np.ndarray:
        transforms.ToPILImage(mode=mode)(torch.from_numpy(tensor)).show()
    elif type(tensor) == torch.Tensor:
        transforms.ToPILImage(mode=mode)(tensor).show()


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def draw_bb(pil_img, left_top=(0,0),  right_bottom=(100, 100)):
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle((left_top, right_bottom), outline="#ff8888", width=2)
    del draw
    pil_img.show()


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def test(images, targets, scale_range=(1.5, 0.7)):
    ## images
    arr_img = np.array(images)
    # show(arr_img)

    w, h, c = arr_img.shape
    scale = [scale_range[1], scale_range[0]]

    # resize: (3, 400, 400)
    resize_h = int(h * scale[0])
    resize_w = int(w * scale[0])
    # img_resize = np.ones((resize_h, resize_w, 3))
    # img_resize[int((resize_h - h) / 2): int((resize_h - h) / 2 + h), int((resize_w - w) / 2): int((resize_w - w) / 2 + w), :] = arr_img
    # img_resize  = img_resize.astype(np.float32)

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
    bottom_right_loc = (
        top_left_loc[0] + crop_h,
        top_left_loc[1] + crop_w
    )

    # crop
    # img_crop = img_resize[top_left_loc[0]: top_left_loc[0] + crop_h, top_left_loc[1]: top_left_loc[1] + crop_w, :]
    padding_size = int((max_crop_size[0] - h) / 2 + 1)
    img_resize = add_margin(images, padding_size, padding_size, padding_size, padding_size, (255, 255, 255))
    img_crop = img_resize.crop((top_left_loc[1], top_left_loc[0], bottom_right_loc[1], bottom_right_loc[0]))

    # resize back: (3, 200, 200)
    # arr_img_resize_back = cv2.resize(img_crop.transpose((1, 2, 0)), (h, w)).transpose((2, 0, 1))
    img_resize_back = img_crop.resize((h, w), resample=Image.BILINEAR)
    arr_img_resize_back = np.array(img_resize_back)
    # img_resize_back = torch.from_numpy(arr_img_resize_back)
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
        if (bb_w / 2)< x_3 < (1 - bb_w / 2) and (bb_h / 2) < y_3 < (1 - bb_h / 2):
        # if 0 < x_3 < 1 and 0 < y_3 < 1:
            list_targets_new.append(target_new)

            # draw bb for testing
            # bb_l_t = ((target_new[2] - target_new[4]/2) * h, (target_new[3] - target_new[5]/2) * w)
            # bb_r_b = ((target_new[2] + target_new[4]/2) * h, (target_new[3] + target_new[5]/2) * w)
            # draw_bb(img_resize_back, bb_l_t, bb_r_b)

    if len(list_targets_new) != 0:
        arr_targets_new = np.array(list_targets_new).astype(np.float32)
        targets_new = torch.from_numpy(arr_targets_new)
        return img_resize_back, targets_new
    else:
        return images, targets


## Rotate
def random_rotate(img, targets, angle_range):
    angle = np.random.randint(angle_range[0], angle_range[1])
    img, targets = rotate_angle(img, targets, angle)
    return Image.fromarray(img), targets


## Crop

## Color
def black_white(img):
    return transforms.Grayscale(3)(img)


def color_jitter(img, dict_args):
    return transforms.ColorJitter(
        brightness=dict_args['b'],
        contrast=dict_args['c'],
        saturation=dict_args['s'],
        hue=dict_args['h']
    )(img)




