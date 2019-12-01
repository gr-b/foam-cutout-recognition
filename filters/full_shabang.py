#! /usr/bin/python

"""
This script demonstrates filtering out everything except the green.
"""

import cv2, sys
import numpy as np
from darkfilter import DarkFilter as df
from lightfilter import LightFilter as lf
from brightnessfilter import IsolateBrightness as ib
from pixelstd import PixelSDT as pix

# demo the brightness filtering
if __name__=='__main__':
    #img = cv2.imread('../testimages/longneck/close/3.jpg')
    #img = cv2.imread('../testimages/longneck/far/175.jpg')
    img = cv2.imread('../../testimages/longneck/far/140.jpg')
    thefilter = ib(50, 175)
    dostd = pix(9.5)
    # brightness filter
    res = thefilter.isolateBrightness(img)
    # std filter
    out = dostd.stdfilter(res)
    print(res.shape)
    # Now, display it
    cv2.imshow('Image',img)
    cv2.imshow('result',out)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image