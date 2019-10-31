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
        print("image shape: ")
        print(image.shape)
        print("Image type: ")
        print(image.dtype)
        # Create mask with same dims but 1 channel
        themask = np.zeros((h,w,1), dtype=image.dtype)
        print("Got here ")
        count = 0
        # stat.stdev(pixel) > self.thresh
        for y in range(0, h):
            for x in range(0, w):
                # image[y, x] is the pixel location
                #count = count + 1
                pixel = image[y, x]
                pi1 = float(pixel[0])
                pi2 = float(pixel[1])
                pi3 = float(pixel[2])
                #print(count)
                #print(pixel)
                output = statistics.stdev((pi1, pi2, pi3))
                #print("STD: "+str(output))
                if output > self.thresh:
                    themask[y, x] = 1
                else:
                    themask[y, x] = 0
                #themask[y, x] = 1 if stat.stdev(pixel) > self.thresh else 0
                #print(count)
        # Now that the mask is created, we can use the mask
        return cv2.bitwise_and(image,image, mask= themask)

# demo the brightness filtering
if __name__=='__main__':
    #img = cv2.imread('../testimages/longneck/close/3.jpg')
    #img = cv2.imread('../testimages/longneck/far/175.jpg')
    img = cv2.imread('../testimages/longneck/far/140.jpg')
    # Try out different float values into the constructor
    thefilter = PixelSDT(9)
    res = thefilter.stdfilter(img)
    print(res.shape)
    # Now, display it
    cv2.imshow('Image',img)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image
