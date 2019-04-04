# PaintingMatcher

## Overview
Image recognition is a task that has been largely tackled in the past few years using various convolutional neural net- work (CNN) architectures. However, paintings are a substantial subset of image data that are slightly more difficult to analyze due to the sometimes abstract nature of paintings and the artistic interpretations made to create them, across artists and painting time periods. To explore possible underlying characteristics of certain paintings and themes, we focus on thematic content of paintings across artists and time periods; specifically, adult portraits, and whether we can successfully classify paintings according to this theme. We use a dataset of over 35K paintings and build on top of VGGNet-16, a successful competitor in the ImageNet challenge, with additional featurizers such as color histograms and Gram matrices. We achieve over 80% classification accuracy with our best model.

## Installation
This project uses Python3.  To create a virtual environment and install all the necessary dependencies, go through the following steps (adapted from Stanford's CS231n [homework setup tutorial](https://cs231n.github.io/assignments2017/assignment1/):

1. *Install Python3*

On a Mac, you can install Python3 by first [installing homebrew](https://brew.sh).  Then, execute `brew install python3` in the Terminal to install Python3.

2. *Install virtualenv*

virtualenv lets you create a "virtual environment" that houses all the necessary requirements for this project, but keeps them isolated from the rest of your system.  To install it, use the Python package manager, `pip`, in the Terminal:
```
sudo pip3 install virtualenv
```

3. *Create a new virtual environment*

Create a virtual environment named `.env` by executing the following command:
```
virtualenv -p python3 .env
```

4. *Working in the virtual environment*

Whenever you would like to work with this project, activate your virtual environment.  You can do this by executing `source .env/bin/activate`.  When you are finished working, you can deactivate your virtual environment by executing `deactivate`.

*Note*: the first time you activate your virtual environment, make sure to install all the necessary requirements for this project.  You can do so by having pip install all packages listed in the requirements file for this project:

```
pip install -r requirements.txt
```

## Dataset




