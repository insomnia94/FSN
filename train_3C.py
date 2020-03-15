# this python script is used to train the 3C network

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

from Config_3C import Config_3C
from Foreground3C import Foreground3C
from DataSet_3C import DataSet_3C

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


  if Config_3C.first_train == True:
    deeplab = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
    deeplab = deeplab.cuda()
    foreground = Foreground3C(deeplab)
  else:
    foreground = torch.load(Config_3C.weight_path)

  foreground = foreground.cuda()

  folder_dataset = generate_folder_dataset(Config_3C.root_image_path, Config_3C.root_label_path)

  transform_image = transforms.Compose([
    transforms.Resize((Config_3C.trans_resize_size, Config_3C.trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
  ])

  transform_label = transforms.Compose([
    transforms.Resize((Config_3C.trans_resize_size, Config_3C.trans_resize_size)),
    transforms.ToTensor(),
  ])

  dataset_3C = DataSet_3C(folder_dataset, transform_image, transform_label)

  train_dataloader = DataLoader(dataset_3C,
                                shuffle=True,
                                num_workers=Config_3C.worker_num,
                                batch_size=Config_3C.batch_num)

  optimizer = optim.Adam(foreground.parameters(), lr=Config_3C.lr)

  criterion = nn.BCELoss(size_average=True)

  start_time = datetime.now()

  for epoch_id in range(Config_3C.epoch_num):
    for iter_id, data in enumerate(train_dataloader, 0):
      images, labels = data
      images = images.cuda()
      labels = labels.cuda()

      foreground.zero_grad()

      fx = foreground(images)

      fx = fx.view(-1)
      y = labels.view(-1)

      seg_loss = criterion(fx, y)
      seg_loss.backward()
      optimizer.step()

      loss_value = seg_loss.detach().cpu().numpy()

      current_time = datetime.now()
      used_time = current_time - start_time

      print(str(epoch_id) + " : " + str(iter_id) + ", loss: " + str(loss_value) + ", time: " + str(used_time))

    if (epoch_id%Config_3C.save_epochs==0) and (epoch_id>0):
      #torch.save(foreground.state_dict(), Config_3C.weight_path)
      torch.save(foreground, Config_3C.weight_path)
      print("model saved")


if __name__ == '__main__':
  main()
