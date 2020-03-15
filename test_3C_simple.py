# this python script is used to test a simple image using 3C foreground segmentation network

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import os
from datetime import datetime
import cv2
import torch.nn.functional as F

from Config_3C import Config_3C
from Foreground3C import Foreground3C
from DataSet_3C import DataSet_3C
from function import *

def main():
  foreground = torch.load(Config_3C.weight_path)
  foreground.eval()
  foreground = foreground.cuda()

  input_path = "./test_images/house.jpg"

  transform_image = transforms.Compose([
    transforms.Resize((Config_3C.trans_resize_size, Config_3C.trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  image = Image.open(input_path)

  w = image._size[0]
  h = image._size[1]

  image = transform_image(image)
  image.unsqueeze_(dim=0)

  image = image.cuda()

  mask = foreground(image)

  mask_binary, _ = pred_resize(mask, h, w)

  cv2.imshow("1", mask_binary)
  cv2.waitKey(0)


if __name__ == '__main__':
  main()