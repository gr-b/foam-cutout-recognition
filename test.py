from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time, os, copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

model = torch.load("model.pt")

folder_name = "test"

data_dir = "./data/"
input_size=224
batch_size=32
num_classes=6

print(model)

# Set up image transforms to resize can crop to 224 by 224
test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # VGG11 takes in normalized values
])

# Set up data dataset and dataloader
dataset   = datasets.ImageFolder(os.path.join(data_dir, folder_name),   test_transforms)
dataloader   = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

running_corrects = 0 # Keep track of the number we get right in testing
start = time.time()
preds = np.array([]) # Array to store the predicted values (for confusion matrix)
labs = np.array([])  # Array to store the real ground truth values (for confusion matrix)
for images, labels in dataloader:
    images_ = images.cuda() # NOTE: Remove this line if not using GPU
    labels_ = labels.cuda() # NOTE: Remove this line if not using GPU

    outputs = model(images_)
    _, predicteds = torch.max(outputs, 1) # max produces the max, and the index of the max
    running_corrects += torch.sum(predicteds == labels_.data) # Update how many we get correct from this batch

    preds = np.append(preds, predicteds.cpu().numpy())
    labs = np.append(labs, labels.cpu().numpy())


acc = running_corrects.double() / len(dataloader.dataset)

elapsed = time.time() - start
print(f"Accuracy: {acc}")
print(f"Correct: {running_corrects} out of {len(dataloader.dataset)}")
print(f"Total test time: {elapsed} | {elapsed / len(dataloader.dataset)} per image")

from sklearn import metrics
print(metrics.confusion_matrix(preds, labs))
