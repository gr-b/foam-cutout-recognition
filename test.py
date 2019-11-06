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

# Default assume that we do NOT have CUDA
dowehavecuda = False

model=None

print("Checking for cuda")
if torch.cuda.is_available():
    print("There is a cuda on this machine")
    # Now we detect if our CUDA version is too old
    try:
        torch.cuda.current_device()
        dowehavecuda = True
        print("We have cuda.")
    except:
        print("Cuda to old. using CPU.")
else:
    print("No cuda found")

# Now we actually open the model
if dowehavecuda:
    # load original model
    # model = torch.load("model-squeezenet.pt")
    # model = torch.load("model.pt")
    checkpoint = torch.load('model.pt')
else:
    # load the cpu version of it
    # model = torch.load("model-squeezenet.pt", map_location=torch.device('cpu'))
    # model = torch.load("model.pt", map_location=torch.device('cpu'))
    checkpoint = torch.load("model.pt", map_location=torch.device('cpu'))
print("Dir command")
print(dir(checkpoint))
print("type command")
print(type(checkpoint))
print("Dict keys")
print(checkpoint.keys())
model.load_state_dict(checkpoint['state_dict'])

"""if torch.cuda.is_available():
    #model = torch.load("model-vgg.pt")
    try:
        torch.cuda.current_device() # this will error out if too old
        #model = torch.load("model-squeezenet.pt")
        model = torch.load("model-squeezenet.pt", map_location=torch.device('gpu'))
    except:
        # If we get here, this means that the cuda version is too old, so we use cpu
        model = torch.load("model-squeezenet.pt", map_location=torch.device('cpu'))
else:
    # We default to CPU version
    model = torch.load("model-squeezenet.pt", map_location=torch.device('cpu'))"""



data_dir = "./data/"
input_size=224
batch_size=1

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
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
# Create training and validation dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val', 'test']}

dataloader = dataloaders_dict['test'] # Get validation dataloader

running_corrects = 0
start = time.time()
for images, labels in dataloader:
    # CUDA check again
    if dowehavecuda:
        images_ = images.cuda()
        labels_ = labels.cuda()
    else:
        images_ = images
        labels_ = labels
    # back to the important part
    outputs = model(images_)
    _, predicteds = torch.max(outputs, 1)
    running_corrects += torch.sum(predicteds == labels_.data)
acc = running_corrects.double() / len(dataloader.dataset)

elapsed = time.time() - start
print(f"Accuracy: {acc}")
print(f"Correct: {running_corrects} out of {len(dataloader.dataset)}")
print(f"Total test time: {elapsed} | {elapsed / len(dataloader.dataset)} per image")
