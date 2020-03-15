# this is dataset python to generate the one pair of image and label for the training for 3C network

from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
from PIL import Image

from Config_3C import Config_3C

class DataSet_3C(Dataset):
  def __init__(self, folder_dataset, transform_image, transform_label):
    self.folder_dataset = folder_dataset
    self.transform_image = transform_image
    self.transform_label = transform_label

  def __getitem__(self, item):
    folder_dataset = self.folder_dataset
    image_path_list = folder_dataset["image"]
    label_path_list = folder_dataset["label"]

    num_samples = len(folder_dataset["image"])
    sample_id = random.randint(0, num_samples-1)

    image_path = image_path_list[sample_id]
    label_path = label_path_list[sample_id]

    image = Image.open(image_path)
    image = self.transform_image(image)

    label = Image.open(label_path)
    label = self.transform_label(label)

    return image, label

  def __len__(self):
    return len(self.folder_dataset["image"])






