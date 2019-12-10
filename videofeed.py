# The hash shabang would go here for python3, but we don't care about that here.
# Import everything necessary
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
import copy, time
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from PIL import Image
import cv2
# font for displaying classification overlay on image feed.
font = cv2.FONT_HERSHEY_SIMPLEX
# Where do we start the text in the image?
org = (20, 50)
# fontScale - do not scale
fontScale = 1
# Blue color in BGR - set the text to the color blue
color = (255, 0, 0) 
# Line thickness of 2 px - this letter thickness is okay
thickness = 2
# Formalize the data configuration
input_size = 224 # The dims of each image being input into the network
# These are the operations done on the image from the image feed such that the network can process the image
# The image is first resized, and then any extra data is removed. The image is then changed into a tensor that Pytorch can recognize (not an opencv numpy array)
# This is essentially creating an object that is used by calling the function after the object is created - kind of like a C++ functor
test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# This is the data structure that maps integers to the classification strings
# This dictionary also matches the order to which the neural network classifiers were ordered in the confusion matrix.
themap = { "0": "empty",
           "1": "fern",
           "2": "bronto",
           "3": "palm",
           "4": "steg",
           "5": "tri", }
# Open the camera on the computer - if this is a laptop, this opens the integrated camera in the monitor. Else, plug in a camera.
cap = cv2.VideoCapture(0)
# load the model
model = torch.load("model.pt")
# extract the weights in the model
model.eval()
# alert user how to run the program.
print("\nTo quit program, press the 'q' key.\n")
while True:
    # The string to print
    stringoutput = "Classification: "
    # read the background
    ret, background = cap.read()
    # convert the datatype from OpenCV coloring to PyTorch coloring
    flip_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    # convert the image to PIL datatype (becomes different underlying C++ type datatype)
    im_pil = Image.fromarray(flip_image)
    # convert the PIL image to a tensor through the image operator created above
    tensorimg = test_transforms(im_pil)
    # This is essentially the python reshape() call, but for tensor objects - reshaping to the 0th dimension.
    tensorimg = tensorimg.unsqueeze(0)
    # Now that the image has been reshaped, it can be passed to the GPU
    tensorimg = tensorimg.cuda()
    # The model can now do the image inference we want - here it spits out a list of inferences
    outputs = model(tensorimg)
    # Now we look at the inference with the largest probability
    _, predicteds = torch.max(outputs, 1)
    # now we tell the user what the item is - but we first have to go from the gpu to the cpu and then grap the data as a numpy array
    result = str(predicteds.cpu().numpy())
    # We now need to extract the part of the string that contains the dictionary index
    result = result.strip('[').strip(']')
    # Grab the classification from the dictionary given the output from the model and append it to the output string.
    stringoutput += str(themap[str(result)])
    # take the camera image feed and overlay the classification string text onto the image.
    background = cv2.putText(background, stringoutput, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
    # Now we show the image stream with the classification showing.
    cv2.imshow('stream',background)
    # Grab the keyboard key
    k = cv2.waitKey(10)
    # Press q to break
    if k == ord('q'):
        break
# Release the camera and destroy all windows         
cap.release()
cv2.destroyAllWindows()
# application is done
