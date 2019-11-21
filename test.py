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

model = torch.load("model-class-weighted.pt")

data_dir = "./data/"
input_size=224
batch_size=32

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
    'test': transforms.Compose([
        transforms.Resize(input_size),
	    transforms.Grayscale(num_output_channels=3), # We do grayscale because green is easy to see
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'hard': transforms.Compose([
        transforms.Resize(input_size),
	    transforms.Grayscale(num_output_channels=3), # We do grayscale because green is easy to see
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test', 'hard']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test', 'hard']}

dataloader = dataloaders_dict['test'] # Get validation dataloader

running_corrects = 0
start = time.time()
preds = np.array([])
labs = np.array([])
for images, labels in dataloader:
    images_ = images.cuda()
    labels_ = labels.cuda()

    outputs = model(images_)
    _, predicteds = torch.max(outputs, 1)
    running_corrects += torch.sum(predicteds == labels_.data)

    preds = np.append(preds, predicteds.cpu().numpy())
    labs = np.append(labs, labels.cpu().numpy())
acc = running_corrects.double() / len(dataloader.dataset)

elapsed = time.time() - start
print(f"Accuracy: {acc}")
print(f"Correct: {running_corrects} out of {len(dataloader.dataset)}")
print(f"Total test time: {elapsed} | {elapsed / len(dataloader.dataset)} per image")

from sklearn import metrics
print(metrics.confusion_matrix(preds, labs))
