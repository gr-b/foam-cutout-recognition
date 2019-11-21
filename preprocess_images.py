# CS549 - Computer Vision Final Project - Deep Learning Approach
# Griffin Bishop, Nick St. George,  Luke Ludington, Andrew Schueler

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

# Path to resize folder
path = "./data/resize/"
# Folder will contain a folder that contains multiple folders for classes.
# resize/   /class1/ /class2/...

batch_size = 30

size = 300

transform = transforms.Compose([
    transforms.Resize(size),
    transforms.CenterCrop(size),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(path, transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

i = 0
for inputs, labels in dataloader:
    for img, label in zip(inputs, labels):
        fp = "out/"+str(label.numpy())+"/"+str(i)+".jpg"
        print(fp)
        torchvision.utils.save_image(img, os.path.join(path, fp))
        i += 1
