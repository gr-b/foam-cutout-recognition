CS549 Computer Vision - Final Project - Deep Learning Approach

To use:

1. Install pytorch and torchvision
2. Get /data/ folder set up with train and validation splits as explained in `train.py`
3. Run `train.py` (specify which model you want to use for pretrained weights [vgg, squeezenet]. This will run for 15 epochs and save the weights to this directory
I've also left `model-squeezenet.pt` pretrained weights in the directory if you don't want to train and just run (since the model is so small).

4. Run `run_model.py` to test.
