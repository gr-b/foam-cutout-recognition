#! /usr/bin/python

"""
This script demonstrates getting the standard deviation of a pixel value.
This means finding the std. dev. of the BGR channel for a single pixel.
This will allow us to remove gray-scale type colors in a COLOR image.

If the standard deviation is larger in magnitude than a threshold, then 
we have found a non-grayscale color, which is what we want.
"""

import cv2, sys
import numpy as np
import statistics
from darkfilter import DarkFilter as df
from lightfilter import LightFilter as lf
from brightnessfilter import IsolateBrightness as ib

class PixelSDT:
    def __init__(self, stdthresh):
        # This is the MINIMUM value that we want for the standard deviation
        self.thresh = float(stdthresh)

    # Creates a binary mask of pixels that exceed the std threshold
    def stdfilter(self, image):
        # Get source image dimensions
        h = image.shape[0]
        w = image.shape[1]
        # Create mask with same dims but 1 channel
        themask = np.zeros((h,w,1), dtype=image.dtype)
        for y in range(0, h):
            for x in range(0, w):
                # image[y, x] is the pixel location
                themask[y, x] = 1 if statistics.stdev((float(image[y, x][0]), float(image[y, x][1]), float(image[y, x][2]))) > self.thresh else 0
        # Now that the mask is created, we can use the mask
        return cv2.bitwise_and(image,image, mask= themask)

# demo the brightness filtering
if __name__=='__main__':
    #img = cv2.imread('../testimages/longneck/close/3.jpg')
    #img = cv2.imread('../testimages/longneck/far/175.jpg')
    #img = cv2.imread('../testimages/longneck/far/120.jpg')
    img = cv2.imread('../../testimages/longneck/close/12.jpg')
    # Try out different float values into the constructor
    #thefilter = PixelSDT(9)
    thefilter = PixelSDT(10)
    res = thefilter.stdfilter(img)
    print(res.shape)
    # Now, display it
    cv2.imshow('Image',img)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
