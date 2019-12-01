#! /usr/bin/python

"""
This script demonstrates filtering out all data that is not the green that we want.

"""

import cv2, sys, os, os.path
import numpy as np
from os import listdir
from os.path import isfile, join
from isolategreen import FindGreen as FindGreen
from darkfilter import DarkFilter as df
from brightnessfilter import IsolateBrightness as bf

# We are going to simply terate through all files in the specified directory and then 
srcdir = "../../testimages/longneck/far/"
# And then save the new files to here
destdir = "../../genimages/longneck/far/"

# check if destination does not exist
if not os.path.exists(destdir):
    # Then create it
    os.makedirs(destdir)
    print("Directory " , destdir ,  " Created ") 

# This lists all of the files in the given directory
#onlyfiles = [f for f in listdir(srcdir) if isfile(join(srcdir, f))]
def findfiles(dir):
    return [f for f in listdir(dir) if isfile(join(dir, f))]

onlyfiles = findfiles(srcdir)
#print(onlyfiles)

# HSV = HSB
# value = brightness
# Hue, saturation, and value
# hue        is from 0 to 360
# saturation is from 0 to 100
# brightness is from 0 to 100
# I used this link to generate these numbers - http://colorizer.org/
#thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([4, 100]) , np.uint8([13, 70]) )
#thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([2, 100]) , np.uint8([13, 70]) )
#thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([2, 100]) , np.uint8([11, 90]) ) # The max value would be 70, but that makes some data go black
thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([2, 100]) , np.uint8([11, 100]) ) # The max value would be 70, but that makes some data go black
# the input is the grayscale magnitude
brightme = bf(25, 240)

# We are going to simply terate through all files in the specified directory and then save the modified files to the desired directory.
for f in listdir(srcdir):
    if isfile(join(srcdir, f)):
        # only if the item is a file do we try and open it as an image
        #img = cv2.imread('../testimages/longneck/far/100.jpg')
        img = cv2.imread(srcdir+str(f))
        # get rid of darks and lights
        res = brightme.isolateBrightness(img)
        # get rid of everything but green
        res = thefilter.isolategreen(res)
        cv2.imwrite(destdir+str(f) , res)

#cv2.imshow('Image',img)
#cv2.imshow('The dark mask',mask)
#cv2.imshow('result',res)
#cv2.waitKey(0) # waits until a key is pressed
#cv2.destroyAllWindows() # destroys the window showing image
