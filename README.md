CS549 Computer Vision - Final Project - Deep Learning Approach

In order to train and test the neural network, run these commands:

1. Install pytorch and torchvision
2. Get /data/ folder set up. Should have /data/train/, /data/val/, /data/test/ folders, each with 6 subfolders (1 for each class)
3. Run `train_multiclass.py`    Note: Requires GPU
4. Run `test.py` to test on the /data/test/ directory.

In order to run the model inference on an image feed with the first detectable camera, run this script:
```
python3 videofeed.py
```
It will then run the classifier on the first camera available. Make sure you have a camera and monitor available before running the script.
