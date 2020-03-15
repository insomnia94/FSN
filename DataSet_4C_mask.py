# this is dataset python to generate the one pair of image and label for the training for 4C network (mask)

from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
from PIL import Image
import cv2

from function import *

class DataSet_4C_mask(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset
    image_path_list = folder_dataset["image"]
    label_path_list = folder_dataset["label"]
    prediction_path_list = folder_dataset["prediction"]

    num_samples = len(folder_dataset["image"])

    while True:

      sample_id = random.randint(0, num_samples-1)

      image_path = image_path_list[sample_id]
      label_path = label_path_list[sample_id]
      prediction_path = prediction_path_list[sample_id]

      image = Image.open(image_path)
      label = Image.open(label_path)
      prediction = Image.open(prediction_path)

      image_numpy = np.asarray(image)
      prediction_numpy = np.asarray(prediction)

      img_h, img_w = image_numpy.shape[:2]
      pred_h, pred_w = prediction_numpy.shape[:2]

      prediction_numpy_resize = cv2.resize(prediction_numpy, (0, 0), fx=img_w / pred_w, fy=img_h / pred_h, interpolation=cv2.INTER_NEAREST)
      prediction_numpy_resize = np.reshape(prediction_numpy_resize, (prediction_numpy_resize.shape[0], prediction_numpy_resize.shape[1], 1))

      input_4C = np.concatenate((image_numpy, prediction_numpy_resize), axis=2)

      input_PIL = Image.fromarray(np.uint8(input_4C))

      input = self.transform_image(input_PIL)
      label = self.transform_label(label)

      return input, label

  def __len__(self):
    return len(self.folder_dataset["image"])