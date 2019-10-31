CS549 Computer Vision - Final Project - Deep Learning Approach

To use:

1. Install pytorch and torchvision
2. Get /data/ folder set up with train and validation splits as explained in `train.py`
3. Run `train.py` (specify which model you want to use for pretrained weights [vgg, squeezenet]. This will run for 15 epochs and save the weights to this directory
I've also left `model-squeezenet.pt` pretrained weights in the directory if you don't want to train and just run (since the model is so small).

4. Run `run_model.py` to test.

# Example filters
There are currently 3 filters included in this repo:
1. dark color filter
2. light color filter
3. standard-dev pixel filter
These filters are used to filter out noise in the image data so as to make the neural network have less data to number crunch.
The dark filter simply removes pixels with a grayscale value smaller than the threshold.
The dark colors, no matter the hue (in HSV), all start to look the same; therefore, we can get rid of those.
THe light filter simply gets rid of colors with a grayscale value larger than the threshold.
This is used because the super bright colors generally do not have an associated hue; rather, they look like a 
grayscale color without having to apply a grayscale conversion. This means that we want to get rid of the colors that
are super bright.

The last filter is a custom filter. All of the image data focuses on green objects. It turns out that the difference in RGB channel
values for the green color we want is considerably larger than their associated grayscale colors. In other words, the integer
values for RGB in the original image aren't close to each other. However, in an image with lots of blacks and whites
in a color image, the channel differences are minimal to none. This means that we can filter out unwanted pixels by setting 
a threshold with the standard deviation of the channel values _for each individual pixel_. This is a super slow process
since we are doing the standard deviation thousands of time, but this is because the filter has not yet been parallelized for a gpu
(OpenGL) or multiple cpu threads. It requires a single threshold parameter as a constructor input. This filter is great because it
removes grayscale colors in an RGB image, therefore doing most of the heavy lifting for a color-searching filter.

I am sure there are easier and faster ways to do pixel channel differencing, but this is what I came up with.

# Other notes
I am also creating a color filter for the green hue that we are wanting to detect. This will be the fourth filter that we can use.
