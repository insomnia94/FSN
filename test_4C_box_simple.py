# this python script is used to test a simple image using 4C foreground segmentation (box) network

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

from Config_4C import Config_4C
from Foreground3C import Foreground3C
from DataSet_3C import DataSet_3C

from function import *

def main():
  foreground = torch.load(Config_4C.box_weight_path)
  foreground.eval()
  foreground = foreground.cuda()

  input_path = "./test_images/house.jpg"

  transform_image = transforms.Compose([
    transforms.Resize((Config_4C.trans_resize_size, Config_4C.trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.4],
                         std=[0.229, 0.224, 0.225, 0.22])
  ])

  image = Image.open(input_path)

  width = image._size[0]
  height = image._size[1]

  image_numpy = np.asarray(image)

  ##############################
  ### generate input, 3C->4C ###

  r = cv2.selectROI(image_numpy)
  x = r[0]
  y = r[1]
  w = r[2]
  h = r[3]

  full_numpy = np.zeros((height, width))
  full_numpy[y:y+h, x:x+w] = 255

  full_numpy = np.reshape(full_numpy, (height, width, 1))

  full_numpy = full_numpy.astype(np.uint8)

  input_4C = np.concatenate((image_numpy, full_numpy), axis=2)

  input_PIL = Image.fromarray(np.uint8(input_4C))
  input = transform_image(input_PIL)
  input.unsqueeze_(dim=0)

  ##################################
  ###########  forward  ############

  input = input.cuda()

  mask = foreground(input)

  mask_binary, _ = pred_resize(mask, height, width)

  cv2.imshow("mask_result", mask_binary)
  cv2.waitKey(0)

if __name__ == '__main__':
  main()