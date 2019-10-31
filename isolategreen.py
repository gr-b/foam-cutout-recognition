#! /usr/bin/python

"""
This script demonstrates filtering out all data that is not the dinosaur.
This is done by color segmentation.
Once the filter properly filters out all but the dino green color, the filter
can be used to generate the data required to create a segmentation network.
"""

import cv2, sys
import numpy as np

class FindGreen:
    def __init__(self, hue, sat, value):
        # We first define all of the constants that we need.
        # Dilation kernel
        self.dilatekernel = np.ones((5,5), np.uint8) 
        # A block Erosion kernel
        self.erodekernel = np.ones((3,3), np.uint8)
        # A Cross Erosion kernel
        self.erode2kernel= np.array([ [ 0, 1, 0] , [ 1, 1, 1] , [ 0, 1, 0]], np.uint8)
        # This is to scale 100 to 255
        scale = float(255/100)
        # scale circle 359 to 179
        circ = float(179/360)
        self.lower_green = np.uint8([ hue[0]*circ , sat[0]*scale , value[0]*scale ])
        self.upper_green = np.uint8([ hue[1]*circ , sat[1]*scale , value[1]*scale ])

    def isolategreen(self, src):
        # First, get the HSV of the BGR image
        hsvimg = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        # Next, we blur a bit
        hsvimg = cv2.medianBlur(hsvimg,3)
        # Now we apply the mask to the image
        mask = cv2.inRange(hsvimg, self.lower_green, self.upper_green)
        # mask = cv2.dilate(mask, erodekernel)
        # mask = cv2.erode(mask, self.erodekernel)
        return cv2.bitwise_and(src,src, mask= mask)

if __name__=='__main__':
    #img = cv2.imread('./longneck/close/3.jpg')
    #img = cv2.imread('../testimages/longneck/far/175.jpg')
    img = cv2.imread('../testimages/longneck/far/100.jpg')
    # HSV = HSB
    # value = brightness
    # Hue, saturation, and value
    # hue        is from 0 to 360
    # saturation is from 0 to 100
    # brightness is from 0 to 100
    # I used this link to generate these numbers - http://colorizer.org/
    thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([4, 100]) , np.uint8([13, 70]) )
    res = thefilter.isolategreen(img)
    cv2.imshow('Image',img)
    #cv2.imshow('The dark mask',mask)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
