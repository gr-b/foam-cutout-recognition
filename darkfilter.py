#! /usr/bin/python

"""
This script demonstrates filtering out all data that is not the dinosaur.
This is done by color segmentation.
Once the filter properly filters out all but the dino green color, the filter
can be used to generate the data required to create a segmentation network.
"""

import cv2, sys
import numpy as np

class DarkFilter:
    def __init__(self, thresh):
        # We first define all of the constants that we need.
        # Dilation kernel
        self.dilatekernel = np.ones((5,5), np.uint8) 
        # A block Erosion kernel
        self.erodekernel = np.ones((3,3), np.uint8)
        # A Cross Erosion kernel
        self.erode2kernel= np.array([ [ 0, 1, 0] , [ 1, 1, 1] , [ 0, 1, 0]], np.uint8)
        # These are the grayscale threshold values
        self.thresh = thresh#50
        self.maxValue = 255
        # Now we can do the filtering

    def doDarkFilter(self, src):
        # First, we 
        th, darkmask = cv2.threshold(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY), self.thresh, self.maxValue, cv2.THRESH_BINARY)
        # Then we erod a bit to get rid of noise.
        darkmask = cv2.erode(darkmask, self.erodekernel) # this looks GOOD!!!
        # Mask the original image with the dark color removing mask
        return cv2.bitwise_and(src,src, mask= darkmask)

if __name__=='__main__':
    #img = cv2.imread('./longneck/close/3.jpg')
    img = cv2.imread('../testimages/longneck/far/175.jpg')
    thefilter = DarkFilter(50)
    res = thefilter.doDarkFilter(img)
    # Now, display it
    cv2.imshow('Image',img)
    #cv2.imshow('The dark mask',darkmask)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
