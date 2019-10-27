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

model = torch.load("model-vgg.pt")

data_dir = "./data/"
input_size=224

print(model)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
	    transforms.Grayscale(num_output_channels=3), # We do grayscale because green is easy to see
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
	    transforms.Grayscale(num_output_channels=3), # We do grayscale because green is easy to see
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, shuffle=True, num_workers=4) for x in ['train', 'val']}

dataloader = dataloaders_dict['val'] # Get validation dataloader

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.features[0].register_forward_hook(get_activation('28'))


for image, label in dataloader:
    image_ = image.cuda()
    label_ = label.cuda()
    start = time.time()
    predicted = model(image_)
    for map in activation['28'].cpu()[0]:
        plt.imshow(map, cmap='gray')
        plt.show()
    #print(predicted, label)
    #plt.imshow(image[0,0])
    #plt.show()
