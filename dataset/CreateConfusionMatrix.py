import argparse
from dataset import *
import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.yticks(np.arange(1), [""])
    plt.xticks(tick_marks, classes, rotation=45)

    print('Confusion matrix, without normalization')
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Painting theme')


def createConfusionMatrix(file):
	if file is None:
		print("Error: missing data file (--dataFile flag)")
		return

	# Read in data
	data = np.load(args.dataFile)
	(_, _, _, _, testInput, testLabels) = loadDatasetRaw()

	themesList = list(set([p.theme for p in testInput]) - set(PORTRAIT_THEMES))
	
	matrix = np.zeros((1, len(themesList)))

	for i in range(len(data)):
		correct = data[i]
		painting = testInput[i]
		if testLabels[i] is not PORTRAIT:
			themeIndex = themesList.index(painting.theme)
			matrix[0][themeIndex] += 1 if correct else -1

	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(matrix, classes=themesList, title='Per-theme grade')
	plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--dataFile', type=str)
args = parser.parse_args()
createConfusionMatrix(args.dataFile)