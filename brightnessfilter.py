#! /usr/bin/python

"""
This script demonstrates filtering out all data that is not the green through
color brightness only.
The color green does not ever reach a certain brightness before becoming a different color.
This script shows how we can remove half of the background without losing ANY 
of the original green color.
"""

import cv2, sys
import numpy as np
from darkfilter import DarkFilter as df
from lightfilter import LightFilter as lf

class IsolateBrightness:
    def __init__(self):
        self.dark  = df()
        self.light = lf()

    def isolateBrightness(self, src):
        # First we remove the dark colors
        brightened = self.dark.doDarkFilter(src)
        # Then we remove the light colors
        #darkened = self.light.doLightFilter(brightened)
        return self.light.doLightFilter(brightened)
        # Now we have the result we want.
        #return darkened

# demo the brightness filtering
if __name__=='__main__':
    #img = cv2.imread('./longneck/close/3.jpg')
    img = cv2.imread('../testimages/longneck/far/175.jpg')
    thefilter = IsolateBrightness()
    res = thefilter.isolateBrightness(img)
    print(res.shape)
    # Now, display it
    cv2.imshow('Image',img)
    cv2.imshow('result',res)
    cv2.waitKey(0) # waits until a key is pressed
    cv2.destroyAllWindows() # destroys the window showing image