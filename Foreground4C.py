# this is the class definition of the foreground segmentation where the input is 4 channels (RGB + X)

import torch.nn as nn

class Foreground4C(nn.Module):
  def __init__(self, deeplab):
    super(Foreground4C, self).__init__()
    self.deeplab = deeplab

    self.convert_conv = nn.Sequential(
      nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
    )

    self.new_classifier = nn.Sequential(
      nn.Conv2d(21, 1, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid(),
    )

    self.up = nn.Sequential(
      nn.Upsample(scale_factor=8, mode="bilinear")
    )

  def forward(self, x):
    x = self.convert_conv(x)
    x = self.deeplab.backbone(x)["out"]
    x = self.deeplab.classifier(x)
    x = self.new_classifier(x)
    x = self.up(x)

    return x