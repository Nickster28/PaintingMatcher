import argparse
from dataset import *
import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt

# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.yticks(np.arange(1), [""])
    plt.xticks(tick_marks, classes, rotation=45)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Painting theme')


def run(file):
	if file is None:
		print("Error: missing data file (--dataFile flag)")
		return

	# Read in data
	data = np.load(args.dataFile)
	(_, _, _, _, testInput, testLabels) = loadDatasetRaw()

	#themesList = list(set([p.theme for p in testInput]) - set(PORTRAIT_THEMES))
	themesList = ['children portraits', 'seas-and-oceans', 'cliffs-and-rocks',
		'female-nude', 'boats-and-ships', 'animals', 'cottages-and-farmhouses']
	
	matrix = np.zeros((1, len(themesList)))
	countMatrix = np.zeros((1, len(themesList)))

	for i in range(len(data)):
		correct = data[i]
		painting = testInput[i]
		if testLabels[i] is not PORTRAIT and painting.theme in themesList:
			themeIndex = themesList.index(painting.theme)
			countMatrix[0][themeIndex] += 1
			if correct:
				matrix[0][themeIndex] += 1

	matrix /= countMatrix
	matrix *= 100.0
	matrix = matrix.round(decimals=2)

	np.set_printoptions(precision=2)
	plt.figure()
	plot_confusion_matrix(matrix, classes=themesList, title='Model Theme Performance (/100)')
	plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--dataFile', type=str)
args = parser.parse_args()
run(args.dataFile)