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
# Formalize the data configuration
input_size = 224
test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# This is the data structure that maps integers to the classification strings
themap = { "0": "empty",
           "1": "fern",
           "2": "bronto",
           "3": "palm",
           "4": "steg",
           "5": "tri", }
# create an overlay image. You can use any image
foreground = np.ones((100,100,3),dtype='uint8')*255
# Open the camera
cap = cv2.VideoCapture(0)
# load the model
model = torch.load("model.pt")
model.eval()
# alert user
print("\nTo quit program, press the 'q' key.\n")
while True:
    # The string to pring
    stringoutput = "Classification: "
    # read the background
    ret, background = cap.read()
    # print("Image type: "+str(background.dtype))
    # convert the datatype
    flip_image = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    # convert the image to PIL
    im_pil = Image.fromarray(flip_image)
    # convert to tensor
    tensorimg = test_transforms(im_pil)
    tensorimg = tensorimg.unsqueeze(0)
    # push to gpu
    tensorimg = tensorimg.cuda()
    # print("I am happy")
    # the classification
    outputs = model(tensorimg)
    _, predicteds = torch.max(outputs, 1)
    # print("Predicted data type: "+str(predicteds.dtype))
    # print("Integer result: "+str(predicteds))
    # now we tell the user what the item is
    result = str(predicteds.cpu().numpy())
    # print("Final now: "+result)
    result = result.strip('[').strip(']')
    # get the number from the string
    # intresult = result
    stringoutput += str(themap[str(result)])
    print(stringoutput)
    #
    cv2.imshow('stream',background)
    k = cv2.waitKey(10)
    # Press q to break
    # if str(k).lower() == ord('q'):
    if k == ord('q'):
        break
# Release the camera and destroy all windows         
cap.release()
cv2.destroyAllWindows()
