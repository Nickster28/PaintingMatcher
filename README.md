# PaintingMatcher

## Overview
Image recognition is a task that has been largely tackled in the past few years using various convolutional neural net- work (CNN) architectures. However, paintings are a substantial subset of image data that are slightly more difficult to analyze due to the sometimes abstract nature of paintings and the artistic interpretations made to create them, across artists and painting time periods. To explore possible underlying characteristics of certain paintings and themes, we focus on thematic content of paintings across artists and time periods; specifically, adult portraits, and whether we can successfully classify paintings according to this theme. We use a dataset of over 35K paintings and build on top of VGGNet-16, a successful competitor in the ImageNet challenge, with additional featurizers such as color histograms and Gram matrices. We achieve over 80% classification accuracy with our best model.

## Installation
This project uses Python2.  To create a virtual environment and install all the necessary dependencies, go through the following steps (adapted from Stanford's CS231n [homework setup tutorial](https://cs231n.github.io/assignments2017/assignment1/):

1. *Install virtualenv*

virtualenv lets you create a "virtual environment" that houses all the necessary requirements for this project, but keeps them isolated from the rest of your system.  To install it, use the Python package manager, `pip`, in the Terminal:
```
sudo pip install virtualenv
```

+ *Create a new virtual environment*

Create a virtual environment named `.env` by executing the following command:
```
virtualenv -p python2 .env
```

+ *Working in the virtual environment*

Whenever you would like to work with this project, activate your virtual environment.  You can do this by executing `source .env/bin/activate`.  When you are finished working, you can deactivate your virtual environment by executing `deactivate`.

*Note*: the first time you activate your virtual environment, make sure to install all the necessary requirements for this project.  You can do so by having pip install all packages listed in the requirements file for this project:

```
pip install -r requirements.txt
```

_Note_: if you have issues installing Pillow, see [this link](https://pillow.readthedocs.io/en/3.0.0/installation.html).

## Dataset
The `dataset.py` file includes information at the top for key functions for how to create and store the dataset of paintings.

## Running
There are several different models used as experiments for this project, which are runnable via `RunModel.py`.  The core model is in `model.py`, which is the generic model that all models extend.  There are several specific model types that build on this.  Each one specifies its variation of the input dataset, what a single row looks like, and what the output from the network should look like.

+ `SimpleResizeModel.py`: A baseline model that is just the VGG architecture with input as resized images

+ `HistogramResizeModel.py`: A model with an additional layer that includes a color histogram

+ `GramResizeModel.py`: A model with an additional layer that includes a Gram matrix

+ `GramHistoResizeModel.py`: A model that includes both a Gram matrix and color histogram




