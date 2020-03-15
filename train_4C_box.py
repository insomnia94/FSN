# the input of the network is the original image (3 channels, RGB) and one channel to indicate the location of the target (box)

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
from collections import OrderedDict
import cv2

from Config_4C import Config_4C
from Foreground4C import Foreground4C
from DataSet_4C_box import DataSet_4C_box

def generate_folder_dataset(root_image_path, root_label_path):
  image_name_list = os.listdir(root_image_path)
  image_name_list.sort()
  image_path_list = []

  for image_name in image_name_list:
    image_path = os.path.join(root_image_path, image_name)
    image_path_list.append(image_path)

  label_name_list = os.listdir(root_label_path)
  label_name_list.sort()
  label_path_list = []

  for label_name in label_name_list:
    label_path = os.path.join(root_label_path, label_name)
    label_path_list.append(label_path)

  folder_dataset = {"image": image_path_list, "label": label_path_list}
  return folder_dataset


def main():

  if Config_4C.first_train == True:
    deeplab = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
    deeplab = deeplab.cuda()
    foreground = Foreground4C(deeplab)
  else:
    foreground = torch.load(Config_4C.box_weight_path)

  foreground = foreground.cuda()

  folder_dataset = generate_folder_dataset(Config_4C.root_image_path, Config_4C.root_label_path)

  transform_image = transforms.Compose([
    transforms.Resize((Config_4C.trans_resize_size, Config_4C.trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.4],
                         std=[0.229, 0.224, 0.225, 0.22])
  ])

  transform_label = transforms.Compose([
    transforms.Resize((Config_4C.trans_resize_size, Config_4C.trans_resize_size)),
    transforms.ToTensor(),
  ])

  dataset_4C = DataSet_4C_box(folder_dataset, transform_image, transform_label)

  train_dataloader = DataLoader(dataset_4C,
                                shuffle=True,
                                num_workers=Config_4C.worker_num,
                                batch_size=Config_4C.batch_num)

  optimizer = optim.Adam(foreground.parameters(), lr=Config_4C.lr)

  criterion = nn.BCELoss(size_average=True)

  start_time = datetime.now()

  for epoch_id in range(Config_4C.epoch_num):
    for iter_id, data in enumerate(train_dataloader, 0):
      inputs, labels = data
      inputs = inputs.cuda()
      labels = labels.cuda()

      foreground.zero_grad()

      fx = foreground(inputs)

      fx = fx.view(-1)
      y = labels.view(-1)

      seg_loss = criterion(fx, y)
      seg_loss.backward()
      optimizer.step()

      loss_value = seg_loss.detach().cpu().numpy()

      current_time = datetime.now()
      used_time = current_time - start_time

      print(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time))

    if (epoch_id%Config_4C.save_epochs==0) and (epoch_id>0):
      #torch.save(foreground.state_dict(), Config_3C.weight_path)
      torch.save(foreground, Config_4C.box_weight_path)
      print("model saved")


if __name__ == '__main__':
  main()