'''
FILE: dataset.py
----------------
This file contains functions to parse, generate, and preprocess the paintings
theme dataset.  The main functions to use are:

* createTrainValTestDatasets: this function reads in the dataset from
dataset.csv and saves to file 3 pickle files: train.pickle, val.pickle, and
test.pickle.  Each pickle file is a list of lists, where each inner list has the
format [PAINTING, PAINTING, SCORE].  Each of the paintings are Painting objects,
and the score is SAME_THEME (1) or DIFFERENT_THEME (0).  There are 120K pairs in
each pickle file.

* loadDatasetRaw: loads in the train, val, and test datasets from the previous
function above to use to run the model.  It loads in the pickle files, fixes
any grayscale issues (see code for more details), and returns the data in the
following format: [train-pairs, train-labels, val-pairs, val-labels, test-pairs, test-labels],
where each pair is in the format [PAINTING, PAINTING] and each label is either
SAME_THEME (1) or DIFFERENT_THEME (0).
----------------
'''

import csv
from collections import defaultdict
import pickle
import random
import progressbar
import itertools
from painting import Painting

# Classification labels for themes
SAME_THEME = 1
DIFFERENT_THEME = 0

    #######################################################################
    ######################## DATASET GENERATION ###########################
    #######################################################################


'''
FUNCTION: parse
---------------
Parameters:
	datasetFile - the CSV filename to parse containing painting data.
					Assumed to be a file in the format of dataset.csv,
					where each row is a painting with the following information: 
					[theme, title, artist, style, genre, wiki url, image url]

Returns: a map from theme name to list of Painting objects in the dataset with
that theme name, randomly shuffled.
---------------
'''
def parse(datasetFile):
	with open(datasetFile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')

		# Skip any paintings without an image URL
		rows = [row for row in reader if row[6].startswith("http")][1:]

		paintings = [Painting(row) for row in rows]
		random.shuffle(paintings)

		paintingsThemeDict = defaultdict(list)
		for painting in paintings:
			paintingsThemeDict[painting.theme].append(painting)

		return paintingsThemeDict

'''
FUNCTION: trimThemesDict
------------------------
Parameters:
	themesDict - a map from theme name to list of Paintings for that theme
	themesToUse - a list of theme names to actually use.

Returns: an updated themesDict containing only themes in themesToUse.
Additionally, each of those theme's painting lists have been trimmed to the
minimum number of paintings among all themesToUse themes.
------------------------
'''
def trimThemesDict(themesDict, themesToUse):
	newThemesDict = {}

	# Trim all theme counts down to the min theme count
	minThemeCount = min([len(themesDict[k]) for k in themesToUse])
	allThemes = themesDict.keys()
	for k in themesToUse:
		newThemesDict[k] = themesDict[k][:minThemeCount]

	return newThemesDict

'''
FUNCTION: addPairFrom
---------------------
Parameters:
	theme1, theme2: the names of the themes to use to select paintings for
					the pair (PAINTING1, PAINTING2).  PAINTING1 will be
					theme1, and PAINTING2 will be theme2.
	themesDict - a dictionary of theme -> list of Paintings of that theme
	datasetPairs - a list of set(painting, painting) containing all existing
				pairs that have been created.
Returns: NA

Picks a random pair of paintings from theme1 and theme2 that have not been
chosen before in datasetPairs, and adds it (as a set) to datasetPairs.
If a pair is chosen, its contained paintings will download their images.
---------------------
'''
def addPairFrom(theme1, theme2, themesDict, datasetPairs):
	theme1Paintings = themesDict[theme1]
	theme2Paintings = themesDict[theme2]
	while True:
		painting1 = theme1Paintings[random.randint(0, 
			len(theme1Paintings) - 1)]
		painting2 = theme2Paintings[random.randint(0, 
			len(theme2Paintings) - 1)]
		pair = set([painting1, painting2])

		if pair not in datasetPairs:

			# Try downloading the paintings' images from the web
			didFinishDownloadingPainting1 = False
			try:
				painting1.downloadImageTo("images")
				didFinishDownloadingPainting1 = True
				painting2.downloadImageTo("images")
				datasetPairs.append(pair)
				break
			except Exception as e:
				print("Exception: " + str(e))
				# Remove the erroring painting from our pairs list
				if didFinishDownloadingPainting1:
					painting1.deleteImageFile("images")
					themesDict[theme2].remove(painting2)
				else:
					themesDict[theme1].remove(painting1)

'''
FUNCTION: generatePairsDataset
------------------------------
Parameters:
	numPairs - the number of dataset pairs to generate.  Must be divisible by
				2*len(themesToUse) and 2*(len(themesToUse) choose 2).
				
	themesToUse - (OPTIONAL) a list of theme names to generate pairs from.  If
					not specified, pairs are generated from all themes.

Returns: a list of sets of 2 paintings

Randomly chooses unique pairs of paintings such that 50% of chosen pairs are
SAME pairs (pairs of paintings with the same theme), and 50% are DIFFERENT
pairs (pairs of paintings with different themes).  Within the 50% of SAME pairs,
pairs are evenly-weighted between each theme.  Within the 50% of DIFFERENT
pairs, pairs are evenly-weighted among all possible combinations of themes.

Relies on a dictionary of paintings categorized into themes, which is read
in from dataset.pickle.  If the file doesn't exist, this function reads in
dataset.csv and creates dataset.pickle from that before proceeding.  Assumed
that dataset.csv has each row as a painting with the following information: 
[theme, title, artist, style, genre, wiki url, image url]
------------------------------
'''
def generatePairsDataset(numPairs, themesToUse=None):

	# Load the dataset from the pickle file or from dataset.csv as a backup
	try:
		themesDict = pickle.load(open("dataset.pickle", "rb"))
	except:
		themesDict = parse('dataset.csv')
		pickle.dump(themesDict, open("dataset.pickle", "wb"))


	# If no themes specified, use all of them
	if not themesToUse:
		themesToUse = themesDict.keys()

	themeCombos = [c for c in itertools.combinations(themesToUse, 2)]
	assert(numPairs % (2 * len(themeCombos)) == 0)
	assert(numPairs % (2 * len(themesToUse)) == 0)

	# Trim down to only themesToUse, where each theme has equal # of paintings
	themesDict = trimThemesDict(themesDict, themesToUse)

	# List of sets of 2 paintings
	datasetPairs = []

	# First pick numPairs / 2 SAME theme pairs, equally from each theme
	numSamePairsPerTheme = numPairs / (2 * len(themesDict.keys()))
	for theme in themesDict:
		bar = progressbar.ProgressBar()
		print("Choosing SAME pairs from " + theme)
		for i in bar(xrange(numSamePairsPerTheme)):
			addPairFrom(theme, theme, themesDict, datasetPairs)

	# Now pick numPairs / 2 DIFFERENT theme pairs, equally from each pairing
	numDiffPairsPerCombo = numPairs / (2 * len(themeCombos))
	for themeCombo in themeCombos:
		bar = progressbar.ProgressBar()
		print("Choosing DIFFERENT pairs from " + str(themeCombo))
		for i in bar(xrange(numDiffPairsPerCombo)):
			addPairFrom(themeCombo[0], themeCombo[1], themesDict, datasetPairs)

	random.shuffle(datasetPairs)
	return datasetPairs

'''
MAIN FUNCTION: createTrainValTestDatasets():
---------------------------------------
Parameters: NA
Returns: NA

Generates 360,000 random unique pairs of paintings from 10 chosen themes, splits
these pairs into thirds, and outputs each third to file as train/val/test data.
The outputted data is a Pickle file serializing a list of lists, each internal
list containing [PAINTING1, PAINTING2, LABEL].
---------------------------------------
'''
def createTrainValTestDatasets():
	# Load the dataset from the pickle file or recreate as a backup
	try:
		dataset = pickle.load(open("trainvaltestdataset.pickle", "rb"))
	except:
		dataset = generatePairsDataset(360000, themesToUse = [
			"female-portraits",
			"male-portraits",
			"forests-and-trees",
			"houses-and-buildings",
			"mountains",
			"boats-and-ships",
			"animals",
			"roads-and-vehicles",
			"fruits-and-vegetables",
			"flowers-and-plants"
		])
		pickle.dump(dataset, open("trainvaltestdataset.pickle", "wb"))

	labeledDataset = []

	# Label each pair
	bar = progressbar.ProgressBar()
	print("Labeling pairs")
	for i in bar(xrange(len(dataset))):
		pair = dataset[i]
		painting1 = pair.pop()

		# Since elements are sets, if a painting is paired with itself the
		# set will be length 1.
		if len(pair) > 0:
			painting2 = pair.pop()
		else:
			painting2 = painting1

		labeledEntry = [painting1, painting2]
		random.shuffle(labeledEntry)
		if painting1.theme == painting2.theme:
			labeledEntry.append(SAME_THEME)
		else:
			labeledEntry.append(DIFFERENT_THEME)

		labeledDataset.append(labeledEntry)

	# Split the pairs into train, val and test
	numPerSubset = len(dataset) / 3

	trainPairs = labeledDataset[: numPerSubset]
	pickle.dump(trainPairs, open("train.pickle", "wb"))

	valPairs = labeledDataset[numPerSubset : 2*numPerSubset]
	pickle.dump(valPairs, open("val.pickle", "wb"))

	testPairs = labeledDataset[2*numPerSubset :]
	pickle.dump(testPairs, open("test.pickle", "wb"))


    #######################################################################
    ########################## DATASET LOADING ############################
    #######################################################################


'''
FUNCTION: loadDatasetRaw
------------------------
Parameters:
	numPairs - the number of pairs to have in the training dataset.  The val/test
				datasets are made to have 1/4 of this number each.
Returns: a list of [train-pairs, train-labels, val-pairs, val-labels, test-pairs, test-labels]
    where each pair entry is a pair of paintings, and each label entry is either
    SAME_THEME or DIFFERENT_THEME.  The painting images have not been modified
	other than converting 1-channel grayscale images to 3-channel RGB images.
	If the pickle files for train/val/test do not exist, returns an empty tuple.
------------------------
'''
def loadDatasetRaw(numPairs):
	# Load the dataset from the pickle files
	train = pickle.load(open("train.pickle", "rb"), encoding='latin1')
	val = pickle.load(open("val.pickle", "rb"), encoding='latin1')
	test = pickle.load(open("test.pickle", "rb"), encoding='latin1')

	# Convert any grayscale images to 3-channel images
	newDatasets = []
	for dataset in [train, val, test]:
		newDataset = []
		newLabels = []
		bar = progressbar.ProgressBar(max_value=len(dataset))
		counter = 0
		for entry in dataset:
			try:
				entry[0].fixGrayscale("images")
				entry[1].fixGrayscale("images")
				newDataset.append(entry[:2])
				newLabels.append(entry[2])
			except Exception as e:
				pass
				
			bar.update(counter)
			counter += 1	

		newDatasets.append(newDataset)
		newDatasets.append(newLabels)	

	newDatasets[0] = newDatasets[0][:numPairs]
	newDatasets[1] = newDatasets[1][:numPairs]
	newDatasets[2] = newDatasets[2][:int(numPairs / 4)]
	newDatasets[3] = newDatasets[3][:int(numPairs / 4)]
	newDatasets[4] = newDatasets[4][:int(numPairs / 4)]
	newDatasets[5] = newDatasets[5][:int(numPairs / 4)]
	
	return newDatasets

