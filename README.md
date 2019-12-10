# CS549 Computer Vision - Final Project - Deep Learning Approach
Griffin Bishop, Luke Ludington, Nick St. George, Andrew Schueler

# Project Description
In this project, teams were tasked with recognizing an object. In our case, it was a collection of 5 foam sticker cutouts. Since they were all the same color and uniform, we decided not to do simple object recognition, as this would be trivial to detect. Instead, we did multi-class classification, where we wanted to distinguish images of the 5 classes of foam cutouts provided. We captured ~3700 images of the cutouts using a robotic turntable in different lighting and distance conditions, and then took ~600 images of them in harder real world environments by hand (for use as the validation set).

We used the VGG11 pre-trained classification network, fine-tuning it on our robotically collected images. Then, we tested our model on the real-world set of images. We achieved an 83.43% accuracy on this set of images. These are favorable results, considering the baseline level of 16.6%.

# Requirements
1. PyTorch
2. Python 3.7
3. Torchvision
4. About 6gb VRAM with current batch size of 24
  (You can use CPU or decrease the batch size, however)

# Usage

1. Use `preprocess_images.py` to crop image resize raw images.
1. Set up /data/. Should have /data/train/, /data/val/, /data/test/ folders, each with 6 subfolders (1 for each class)
2. Run `train_multiclass.py`    Note: Comment out certain lines if not using GPU with cuda
    This will write a "model.pt" to this directory.
3. Run `test.py` to test on the /data/test/ directory.

In order to run the model inference on an image feed with the first detectable camera, run this script:
```
python3 videofeed.py
```
It will then run the classifier on the first camera available. Make sure you have a camera and monitor available before running the script.
