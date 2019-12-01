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

import cv2
 
# create an overlay image. You can use any image
foreground = np.ones((100,100,3),dtype='uint8')*255
# Open the camera
cap = cv2.VideoCapture(0)
# load the model
model = torch.load("model.pt")
model.eval()
# alert user
print("\nTo quit program, press the 'q' button.\n")
while True:
    # read the background
    ret, background = cap.read()
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
