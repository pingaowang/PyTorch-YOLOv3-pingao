import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms


def show(tensor, mode='RGB'):
    transforms.ToPILImage(mode=mode)(tensor).show()



def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


## Crop

## Color






