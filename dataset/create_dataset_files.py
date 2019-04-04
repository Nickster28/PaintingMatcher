'''
This file runs the logic to create train/test/val pickle files for use to train the Painting model.
It creates a dataset out of the dataset.csv file, downloads the image files from the internet for
the images used in the dataset, and creates train.pickle/val.pickle/test.pickle files used in later
parts of the model.
'''

from dataset import *
createTrainValTestDatasets()