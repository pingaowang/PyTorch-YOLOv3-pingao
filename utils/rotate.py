import numpy as np
import cv2
import torch
from PIL import Image


def show(img):
    Image.fromarray(img).show()


def label2coord(label,
                height_image,
                width_image):
    category      =       label[0]
    x_center_bbox = float(label[1])
    y_center_bbox = float(label[2])
    width_bbox    = float(label[3])
    height_bbox   = float(label[4])
    x_left   = int( (x_center_bbox- width_bbox/2.) * width_image )
    x_right  = int( (x_center_bbox+ width_bbox/2.) * width_image )
    y_top    = int( (y_center_bbox-height_bbox/2.) * height_image )
    y_bottom = int( (y_center_bbox+height_bbox/2.) * height_image )
    return category, x_left, y_top, x_right, y_bottom


def coord2label(coord,
                height_image,
                width_image):
    category =       coord[0]
    x_left   = float(coord[1])
    y_top    = float(coord[2])
    x_right  = float(coord[3])
    y_bottom = float(coord[4])
    x_center_bbox = (x_left  +x_right )/2. / width_image
    y_center_bbox = (y_top   +y_bottom)/2. / height_image
    width_bbox    = (x_right -x_left  )    / width_image
    height_bbox   = (y_bottom-y_top   )    / height_image
    return 0, category, x_center_bbox, y_center_bbox, width_bbox, height_bbox


def rotate_angle(pil_img, labels0, angle):
    # print("====")
    # pil_img.show()
    # print(labels0)
    image0 = np.array(pil_img)
    height_image0, width_image0 = image0.shape[:2]
    coords = []
    l_arr_label = list()
    for tensor_label in labels0:
        arr_label = tensor_label.numpy()
        label = "{} {} {} {} {}".format(
            str(arr_label[1]),
            str(arr_label[2]),
            str(arr_label[3]),
            str(arr_label[4]),
            str(arr_label[5])
        )
        label = label.split()
        coord = label2coord(label,height_image0,width_image0)
        h2 = 2*height_image0
        w2 = 2*width_image0
        coords.append([coord[0],    coord[1],    coord[2],    coord[3],    coord[4]])
        # 4 copies for reflections
        coords.append([coord[0],   -coord[3],    coord[2],   -coord[1],    coord[4]])
        coords.append([coord[0], w2-coord[3],    coord[2], w2-coord[1],    coord[4]])
        coords.append([coord[0],    coord[1],   -coord[4],    coord[3],   -coord[2]])
        coords.append([coord[0],    coord[1], h2-coord[4],    coord[3], h2-coord[2]])

    if angle == 0:
        image = image0.copy()
    else:
        center = int(width_image0/2), int(height_image0/2)
        scale = 1.
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        #image = cv2.warpAffine(image0,matrix,(width_image0,height_image0),borderMode=cv2.BORDER_REPLICATE)
        image = cv2.warpAffine(image0,matrix,(width_image0,height_image0),borderMode=cv2.BORDER_REFLECT_101)

    for coord in coords:
        category, x_left0, y_top0, x_right0, y_bottom0 = coord
        area0 = (x_right0-x_left0)*(y_bottom0-y_top0)
        if angle == 0:
            x_left, y_top, x_right, y_bottom = x_left0, y_top0, x_right0, y_bottom0
        else:
            points0 = np.array([[x_left0 , y_top0   , 1.],
                                [x_left0 , y_bottom0, 1.],
                                [x_right0, y_top0   , 1.],
                                [x_right0, y_bottom0, 1.]])
            points = np.dot( matrix , points0.T ).T
            x_left   = int(min( p[0] for p in points ))
            x_right  = int(max( p[0] for p in points ))
            y_top    = int(min( p[1] for p in points ))
            y_bottom = int(max( p[1] for p in points ))
        x_left, x_right = np.clip( [x_left, x_right] , 0, width_image0 )
        y_top, y_bottom = np.clip( [y_top, y_bottom] , 0, height_image0 )

        area = (x_right - x_left) * (y_bottom - y_top)
        if area > area0 * 0.8:
            label = coord2label([category, x_left, y_top, x_right, y_bottom], height_image0, width_image0)
            arr_label = np.array(label).astype(np.float)
            l_arr_label.append(arr_label)

    arr_label_out = None
    for label in l_arr_label:
        # filter of bad b-box
        if label[3] * label[4] != 0:
            # if (label[1] + label[3] / 2 < 1) and (label[1] - label[3] / 2 > 0) and (label[2] + label[4] / 2 < 1) and (label[2] - label[4] / 2 > 0):
            #     print("No:")
            #     print(label)
            # else:
            #     print("Yes:")
            #     print(label)
            if arr_label_out is not None:
                arr_label_out = np.concatenate((arr_label_out, [label]), axis=0)

            else:
                arr_label_out = np.array([label])
    # print(arr_label_out)
    if arr_label_out is not None:
        tensor_label_out = torch.from_numpy(arr_label_out)
    else:
        tensor_label_out = torch.zeros((1, 6))
    # print(tensor_label_out)
    # show(image)
    # print("=====")

    # tensor_label_out = None
    return image, tensor_label_out




