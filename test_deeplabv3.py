# this script is used to test the original pytorch deeplab v3 model (semantic)

import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import urllib

model = torch.hub.load('pytorch/vision:v0.4.2', 'deeplabv3_resnet101', pretrained=True)
model.eval()

resnet50 = models.resnet50(pretrained=True)
resnet50.eval()

image_path = "./test_images/bus.jpg"

input_image = Image.open(image_path)

image_numpy = np.asanyarray(input_image)/255

preprocess = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)

input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():

  # get the output step by step
  output_backbone = model.backbone(input_batch)["out"]
  output_classifier = model.classifier(output_backbone)
  output_upsample = F.upsample(output_classifier, scale_factor=8, mode="bilinear")
  output = output_upsample.view([21, 448, 448])

  #original way to get the output
  #output = model(input_batch)['out'][0]

output_predictions = output.argmax(0)

# create a color pallette, selecting a color for each class
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)

import matplotlib.pyplot as plt
plt.imshow(r)
plt.show()

pass