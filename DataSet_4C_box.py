# this is dataset python to generate the one pair of image and label for the training for 4C network (box)

from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
from PIL import Image
import cv2

from function import *

class DataSet_4C_box(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset
    image_path_list = folder_dataset["image"]
    label_path_list = folder_dataset["label"]

    num_samples = len(folder_dataset["image"])

    while True:

      sample_id = random.randint(0, num_samples-1)

      image_path = image_path_list[sample_id]
      label_path = label_path_list[sample_id]

      image = Image.open(image_path)
      label = Image.open(label_path)

      image_numpy = np.asarray(image)
      label_numpy = np.asarray(label)

      width = label_numpy.shape[1]
      height = label_numpy.shape[0]

      box_list = mask2box(label_numpy)

      if len(box_list) > 0:
        box_list = box_list[0]

        x0 = box_list[0]
        y0 = box_list[1]
        x1 = box_list[2]
        y1 = box_list[3]

        full_numpy = np.zeros((height, width))
        full_numpy[y0:y1, x0:x1] = 255

        full_numpy = np.reshape(full_numpy, (height, width, 1))

        full_numpy = full_numpy.astype(np.uint8)

        input_4C = np.concatenate((image_numpy, full_numpy), axis=2)

        input_PIL = Image.fromarray(np.uint8(input_4C))

        input = self.transform_image(input_PIL)
        label = self.transform_label(label)

        return input, label

  def __len__(self):
    return len(self.folder_dataset["image"])

