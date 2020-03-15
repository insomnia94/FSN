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
import scipy.stats

# final J_m (RL & Base)
#x=[0.954, 0.555, 0.802, 0.938, 0.971, 0.961, 0.946, 0.873, 0.951, 0.891, 0.927, 0.912, 0.873, 0.667, 0.861, 0.904, 0.625, 0.918, 0.898, 0.901]
#y=[0.954, 0.525, 0.684, 0.840, 0.971, 0.960, 0.946, 0.846, 0.951, 0.872, 0.913, 0.911, 0.868, 0.667, 0.861, 0.864, 0.625, 0.914, 0.890, 0.862]

# IOU_B (RL & Base)
#x=[99.4, 98.9, 97.4, 98.8, 99.4, 99.7, 99.1, 98.1, 99.3, 99.6, 99.1, 99.0, 98.9, 98.3, 99.3, 93.9, 97.1, 99.4, 98.6, 98.3]
#y=[44.6, 75.1, 55.9, 45.0, 41.8, 60.1, 43.1, 59.3, 52.6, 72.1, 64.4, 58.4, 61.8, 72.4, 66.3, 41.2, 69.1, 67.3, 58.2, 46.1]

# IOU_F (RL & Base)
x=[91.0, 39.9, 74.9, 89.9, 95.2, 92.7, 91.6, 83.3, 92.6, 83.0, 90.1, 88.8, 80.0, 57.2, 74.9, 87.5, 58.7, 85.1, 86.1, 86.0]
y=[74.6, 15.6, 55.0, 70.8, 85.4, 79.2, 76.9, 63.2, 78.9, 55.6, 75.8, 73.7, 52.4, 28.0, 45.1, 70.7, 44.9, 61.7, 68.1, 65.4]

result = scipy.stats.ranksums(x, y)

pass


