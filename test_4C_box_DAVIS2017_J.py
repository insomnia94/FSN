# evaluate the accuracy on DAVIS 2017 dataset using Foreground Segmentation 4C network (box)

import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from datetime import datetime
import cv2

from Config_4C import Config_4C
from Foreground4C import Foreground4C
from function import *

def main():

  root_images_path = Config_4C.test_root_images_path
  root_labels_mask_path = Config_4C.test_root_labels_mask_path
  root_labels_box_small_path = Config_4C.test_root_labels_box_small_path
  root_labels_box_margin_path = Config_4C.test_root_labels_box_margin_path

  result_path = Config_4C.result_path

  # initialize the paths of all images, annotations (mask, box)

  image_target_frame_path_list = []
  label_box_small_target_frame_path_list = []
  label_box_margin_target_frame_path_list = []
  label_mask_target_frame_path_list = []

  target_name_list = os.listdir(root_images_path)
  target_name_list.sort()

  if Config_4C.sequence_only_test == True:
    target_name_list = Config_4C.test_sequence_list

  for target_name_id in range(len(target_name_list)):

    image_target_frame_path_list.append([])
    label_mask_target_frame_path_list.append([])
    label_box_small_target_frame_path_list.append([])
    label_box_margin_target_frame_path_list.append([])

    image_target_path = os.path.join(root_images_path, target_name_list[target_name_id])
    label_mask_target_path = os.path.join(root_labels_mask_path, target_name_list[target_name_id])
    label_box_small_target_path = os.path.join(root_labels_box_small_path, target_name_list[target_name_id])
    label_box_margin_target_path = os.path.join(root_labels_box_margin_path, target_name_list[target_name_id])

    frame_name_list = os.listdir(image_target_path)
    frame_name_list.sort()

    for frame_name_id in range(len(frame_name_list)):
      frame_name = frame_name_list[frame_name_id].split(".")[0]
      frame_name_image = frame_name + ".jpg"
      frame_name_mask  = frame_name + '.png'
      frame_name_box = frame_name + ".txt"

      image_frame_path = os.path.join(image_target_path, frame_name_image)
      label_mask_frame_path = os.path.join(label_mask_target_path, frame_name_mask)
      label_box_small_frame_path = os.path.join(label_box_small_target_path, frame_name_box)
      label_box_margin_frame_path = os.path.join(label_box_margin_target_path, frame_name_box)

      image_target_frame_path_list[target_name_id].append(image_frame_path)
      label_mask_target_frame_path_list[target_name_id].append(label_mask_frame_path)
      label_box_small_target_frame_path_list[target_name_id].append(label_box_small_frame_path)
      label_box_margin_target_frame_path_list[target_name_id].append(label_box_margin_frame_path)

  # initialize the network
  foreground = torch.load(Config_4C.box_weight_path)
  foreground.eval()
  foreground = foreground.cuda()

  transform_image = transforms.Compose([
    transforms.Resize((Config_4C.trans_resize_size, Config_4C.trans_resize_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.4],
                         std=[0.229, 0.224, 0.225, 0.22])
  ])

  # evaluation starts here

  target_name_list = os.listdir(root_images_path)
  target_name_list.sort()

  if Config_4C.sequence_only_test == True:
    target_name_list = Config_4C.test_sequence_list

  # to calculate the overall average accuracy
  accuracy_target_sum = 0

  for target_name_id in range(len(target_name_list)):

    # to calculate the average accuracy of a certain target
    accuracy_frame_sum = 0

    result_target_path = os.path.join(result_path, target_name_list[target_name_id])
    if not os.path.isdir(result_target_path):
      os.mkdir(result_target_path)

    image_target_path = os.path.join(root_images_path, target_name_list[target_name_id])
    frame_name_list = os.listdir(image_target_path)
    frame_name_list.sort()

    for frame_name_id in range(len(frame_name_list)):
      frame_name = frame_name_list[frame_name_id].split(".")[0]
      result_frame_path = os.path.join(result_target_path, frame_name + ".png")

      image_path = image_target_frame_path_list[target_name_id][frame_name_id]
      label_mask_path = label_mask_target_frame_path_list[target_name_id][frame_name_id]
      label_box_small_path = label_box_small_target_frame_path_list[target_name_id][frame_name_id]
      label_box_margin_path = label_box_margin_target_frame_path_list[target_name_id][frame_name_id]

      # import the image
      image = Image.open(image_path)

      # the width and height of the intact image
      w_intact = image._size[0]
      h_intact = image._size[1]

      # import the mask label
      label_mask = cv2.imread(label_mask_path, cv2.IMREAD_UNCHANGED)

      # import the box small_label
      f = open(label_box_small_path, "r")
      box_small_list = f.read().splitlines()

      # import the box margin_label
      f = open(label_box_margin_path, "r")
      box_margin_list = f.read().splitlines()

      # ignore the situation where the target is missing
      if len(box_margin_list) == 4:
        margin_x0 = int(box_margin_list[0])
        margin_y0 = int(box_margin_list[1])
        margin_x1 = int(box_margin_list[2])
        margin_y1 = int(box_margin_list[3])

        small_x0 = int(box_small_list[0])
        small_y0 = int(box_small_list[1])
        small_x1 = int(box_small_list[2])
        small_y1 = int(box_small_list[3])

        # crop the image
        image_margin = image.crop((margin_x0, margin_y0, margin_x1, margin_y1))
        image_margin_numpy = np.asarray(image_margin)

        # the width and height of the image patch using box margin
        w_margin = image_margin._size[0]
        h_margin = image_margin._size[1]

        # the width and height of the image patch using box small
        w_small = small_x1 - small_x0
        h_small = small_y1 - small_y0

        # the new coordinate of the small box
        new_small_x0 = small_x0 - margin_x0
        new_small_y0 = small_y0 - margin_y0
        new_small_x1 = small_x1 - margin_x0
        new_small_y1 = small_y1 - margin_y0

        full_numpy = np.zeros((h_margin, w_margin))
        full_numpy[new_small_y0:new_small_y1, new_small_x0:new_small_x1] = 255
        full_numpy = np.reshape(full_numpy, (h_margin, w_margin, 1))
        full_numpy = full_numpy.astype(np.uint8)

        input_4C = np.concatenate((image_margin_numpy, full_numpy), axis=2)
        input_PIL = Image.fromarray(np.uint8(input_4C))
        image_trans = transform_image(input_PIL)

        # pre-process the image patch
        #image_trans = transform_image(image_margin)
        image_trans.unsqueeze_(dim=0)
        image_trans = image_trans.cuda()

        # generate the prediction
        pred_mask = foreground(image_trans)

        pred_mask_resize, _ = pred_resize(pred_mask, h_margin, w_margin)

        final_pred_mask = np.zeros((h_intact, w_intact))
        final_pred_mask[margin_y0:margin_y1, margin_x0:margin_x1] = pred_mask_resize

        accuracy = mask_iou(final_pred_mask, label_mask)

        accuracy_frame_sum += accuracy

        if Config_4C.save_result_png == True:
          cv2.imwrite(result_frame_path, final_pred_mask)
      else:
        accuracy_frame_sum += 1
        if Config_4C.save_result_png == True:
          cv2.imwrite(result_frame_path, label_mask)

      '''
      # only for test
      cv2.imshow("result", final_pred_mask)
      cv2.imshow("label", label_mask)
      cv2.waitKey(0)
      '''

    target_average_accuracy = accuracy_frame_sum / (frame_name_id+1)
    print(target_name_list[target_name_id] + ": " + str(round(target_average_accuracy, 3)))

    accuracy_target_sum += target_average_accuracy

  overall_average_accuracy = accuracy_target_sum / (target_name_id+1)

  print()
  print("overall accuracy: " + str(round(overall_average_accuracy, 3)))




if __name__ == '__main__':
  main()
