#! /usr/bin/python

"""
This script demonstrates filtering out all data that is not the dinosaur.
This is done by color segmentation.
Once the filter properly filters out all but the dino green color, the filter
can be used to generate the data required to create a segmentation network.
"""

import cv2, sys
import numpy as np

class LightFilter:
    def __init__(self):
        # We first define all of the constants that we need.
        # Dilation kernel
        self.dilatekernel = np.ones((5,5), np.uint8) 
        # A block Erosion kernel
        self.erodekernel = np.ones((3,3), np.uint8)
        # A Cross Erosion kernel
        self.erode2kernel= np.array([ [ 0, 1, 0] , [ 1, 1, 1] , [ 0, 1, 0]], np.uint8)
        # These are the grayscale threshold values - we could go more risky with lower upper threshold values,
        # but we don't want to risk it.
        #self.thresh = 175
        self.thresh = 180
        self.maxValue = 255
        # Now we can do the filtering

    def doLightFilter(self, src):
        # First, we 
        th, darkmask = cv2.threshold(cv2.cvtColor(src,cv2.COLOR_BGR2GRAY), self.thresh, self.maxValue, cv2.THRESH_BINARY)
        # Then we erod a bit to get rid of noise.
        darkmask = cv2.erode(darkmask, self.erodekernel) # this looks GOOD!!!
        # Then we invert to keep the darker color
        lightmask = cv2.bitwise_not(darkmask)
        # Mask the original image with the light color removing mask
        return cv2.bitwise_and(src,src, mask= lightmask)
        #return cv2.bitwise_and(src,src, mask= lightmask) , lightmask

if __name__=='__main__':
    #img = cv2.imread('./longneck/close/3.jpg')
    img = cv2.imread('../testimages/longneck/far/175.jpg')
    thefilter = LightFilter()
    #res , mask= thefilter.doLightFilter(img)
    res = thefilter.doLightFilter(img)
    # Now, display it
    cv2.imshow('Image',img)
    #cv2.imshow('The light mask',mask)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
