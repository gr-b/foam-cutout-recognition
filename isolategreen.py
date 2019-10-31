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
        # This is the green we want to find
        # self.findgreen = greenBGR
        # Green color converted to HSV
        # self.hsv_green = cv2.cvtColor(self.findgreen,cv2.COLOR_BGR2HSV)
        # We create a delta value for the hue
        # self.greendelta = thresh
        # The bounds of the HSV green filter
        # Now we create the "lower" and "upper" values of the HSV filter
        # self.lower_green = self.hsv_green - np.uint8([ self.greendelta , 0 , 0 ])
        # self.upper_green = self.hsv_green + np.uint8([ self.greendelta , 0 , 0 ])
        #
        # This is to scale 100 to 255
        scale = float(255/100)
        # scale circle 359 to 179
        circ = float(179/360)
        #self.lower_green = np.uint8([ hue[0]*circ , sat[0]*scale , value[0]*scale ])
        #self.upper_green = np.uint8([ hue[1]*circ , sat[1]*scale , value[1]*scale ])
        self.lower_green = np.uint8([ hue[0]*circ , 0 , value[0]*scale ])
        self.upper_green = np.uint8([ hue[1]*circ , 255 , value[1]*scale ])
        # For mask creation
        self.maxValue = 255

    def isolategreen(self, src):
        # First, get the HSV of the BGR image
        hsvimg = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
        # Next, we blur a bit
        hsvimg = cv2.medianBlur(hsvimg,3)
        # Now we apply the mask to the image
        mask = cv2.inRange(hsvimg, self.lower_green, self.upper_green)
        # Get rid of noise in the mask
        # mask = cv2.dilate(mask, self.dilatekernel)
        # mask = cv2.dilate(mask, self.dilatekernel)
        #mask = cv2.dilate(mask, erodekernel)
        # mask = cv2.erode(mask, self.erodekernel)
        # mask = cv2.erode(mask, self.erodekernel)
        # mask = cv2.erode(mask, self.erodekernel)
        # mask = cv2.erode(mask, self.erodekernel)
        # One last dilation
        # mask = cv2.dilate(mask, self.erode2kernel)
        # mask = cv2.dilate(mask, self.erode2kernel)
        # invert the green finding mask
        # mask = cv2.bitwise_not(mask)

        #th, darkmask = cv2.threshold(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY), self.thresh, self.maxValue, cv2.THRESH_BINARY)
        # Then we erod a bit to get rid of noise.
        #darkmask = cv2.erode(darkmask, self.erodekernel) # this looks GOOD!!!
        # Mask the original image with the dark color removing mask
        #return cv2.bitwise_and(src,src, mask= mask) , mask
        return cv2.bitwise_and(src,src, mask= mask)

if __name__=='__main__':
    #img = cv2.imread('./longneck/close/3.jpg')
    img = cv2.imread('../testimages/longneck/far/175.jpg')
    img = cv2.imread('../testimages/longneck/far/100.jpg')
    #thefilter = FindGreen(np.uint8([[[140,185,130]]]) , 56)
    # HSV = HSB
    # value = brightness
    # Hue, saturation, and value
    # hue        is from 0 to 360
    # saturation is from 0 to 100
    # brightness is from 0 to 100
    # I used this link to generate these numbers
    #thefilter = FindGreen(np.uint8([100, 130]) , np.uint8([55, 100]) , np.uint8([50, 100]) )
    #thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([55, 100]) , np.uint8([50, 100]) )
    thefilter = FindGreen(np.uint8([125, 162]) , np.uint8([55, 100]) , np.uint8([13, 70]) )
    #res , mask = thefilter.isolategreen(img)
    res = thefilter.isolategreen(img)
    cv2.imshow('Image',img)
    #cv2.imshow('The dark mask',mask)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
