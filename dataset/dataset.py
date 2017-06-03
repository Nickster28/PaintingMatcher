import csv
import urllib
from collections import defaultdict
import pickle
import random
import progressbar
import os
import itertools
from PIL import Image
from scipy import misc

# Classification labels for themes
SAME_THEME = 1
DIFFERENT_THEME = 0


'''
CLASS: Painting
----------------
A class representing a single painting.  A painting has the following data
associated with it:
	- theme (e.g. "musical-instruments")
	- title (e.g. "A Turkish Girl")
	- artist (e.g. "Karl Bryullov")
	- style (e.g. "Romanticism")
	- genre (e.g. "portrait")
	- wiki URL
	- image URL
----------------
'''
class Painting:

	# A static counter that assigns a unique id to each created painting
	paintingID = 0

	'''
	init
	--------------
	Parameters:
		dataList - a length-7 list containing the row of data for the painting
		to be created.  The row should have the format:

		[theme, title, artist, style, genre, wiki url, image url]
	--------------
	'''
	def __init__(self, dataList):
		self.id = Painting.paintingID
		Painting.paintingID += 1

		self.theme = dataList[0]
		self.title = dataList[1]
		self.artist = dataList[2]
		self.style = dataList[3]
		self.genre = dataList[4]
		self.wikiURL = dataList[5]
		self.imageURL = dataList[6]

	'''
	METHOD: imageFilename
	---------------------
	Parameters: NA
	Returns: the name of the image file to use for this painting.
	---------------------
	'''
	def imageFilename(self):
		return str(self.id) + ".jpg"

	'''
	METHOD: downloadImageTo
	---------------------
	Parameters:
		directory - the directory in which to save this image

	Returns: NA

	Attempts to download the image for this painting into the given directory if
	it has not been downloaded to this location already (in which case we do
	nothing).  The image will be saved as ID.jpg, where ID is the ID of this
	painting.  If the download fails, an exception is thrown.
	---------------------
	'''
	def downloadImageTo(self, directory):
		fullFilename = directory + "/" + self.imageFilename()
		if not os.path.isfile(fullFilename):
			urllib.urlretrieve(self.imageURL, fullFilename)

	'''
	METHOD: deleteImageFile
	-----------------------
	Parameters:
		directory - the directory in which to delete this painting's image.
	Returns: NA

	If the file exists, deletes our image file on disk in the given directory.
	-----------------------
	'''
	def deleteImageFile(self, directory):
		fullFilename = directory + "/" + self.imageFilename()
		if os.path.isfile(fullFilename):
			os.remove(fullFilename)

	'''
	METHOD: repr
	------------
	Parameters: NA
	Returns: a string representation of this painting of the format 
		ID,FILENAME,THEME,TITLE,ARTIST,STYLE,GENRE,WIKIURL,IMAGEURL
	------------
	'''
	def __repr__(self):
		return ",".join([str(self.id), self.imageFilename(), self.theme,
			self.title, self.artist, self.style, self.genre, self.wikiURL,
			self.imageURL])


'''
FUNCTION: parse
---------------
Parameters:
	datasetFile - the filename to parse containing painting data

Returns: a map from theme name to list of Painting objects in the dataset with
that theme name.
---------------
'''
def parse(datasetFile):
	with open(datasetFile, 'rb') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
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
------------------------------
'''
def generatePairsDataset(numPairs, themesToUse=None):

	# If no themes specified, use all of them
	if not themesToUse:
		themesToUse = themesDict.keys()

	themeCombos = [c for c in itertools.combinations(themesToUse, 2)]
	assert(numPairs % (2 * len(themeCombos)) == 0)
	assert(numPairs % (2 * len(themesToUse)) == 0)

	# Load the dataset from the pickle file or from dataset.csv as a backup
	try:
		themesDict = pickle.load(open("dataset.pickle", "rb"))
	except:
		themesDict = parse('dataset.csv')
		pickle.dump(themesDict, open("dataset.pickle", "wb"))

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

	labeledDataset = []

	# Label each pair
	for i in xrange(len(dataset)):
		labeledEntry = list(dataset[i])
		if labeledEntry[0].theme == labeledEntry[1].theme:
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

'''
FUNCTION: loadDatasetRaw
------------------------
Parameters: NA
Returns: a (train, val, test) tuple where each entry is a list of
	[painting, painting, score].  The painting images have not been modified
	other than converting 1-channel grayscale images to 3-channel RGB images.
	If the pickle files for train/val/test do not exist, returns an empty tuple.
------------------------
'''
def loadDatasetRaw():
	try:
		# Load the dataset from the pickle files
		train = pickle.load(open("train.pickle", "rb"))
		val = pickle.load(open("val.pickle", "rb"))
		test = pickle.load(open("test.pickle", "rb"))

		# Convert any grayscale images to 3-channel images
		for dataset in [train, val, test]:
			for entry in dataset:
				for painting in entry[:2]:
					image = misc.imread("images/" + painting.imageFilename())
					if len(image.shape) != 3:
					    newImage = np.zeros(image.shape + (3,))
					    for row in range(image.shape[0]):
					        for col in range(image.shape[1]):
					            newImage[row][col][0] = image[row][col]
					            newImage[row][col][1] = image[row][col]
					            newImage[row][col][2] = image[row][col]
					    misc.imsave("images/" + painting.imageFilename(), newImage)

		return (train, val, test)
	except:
		return ()

'''
FUNCTION: loadDatasetResize
---------------------------
Parameters:
	width, height - the dimensions to resize all painting images to
	imagesDirectory - the name of the directory in which to save the resized
					versions of images.

Returns: a (train, val, test) tuple where each entry is a list of 
	[painting, painting, score].  The painting images have been modified and
	saved in the given directory to be resized to the given dimensions.
---------------------------
'''
def loadDatasetResize(width, height, imagesDirectory):
	datasets = loadDatasetRaw()
	for dataset in datasets:
		for entry in dataset:
			for painting in entry[:2]:
				img = Image.open("images/" + painting.imageFilename())
				img = img.resize((width, height))
				img.save(imagesDirectory + "/" + painting.imageFilename())

	return datasets

'''
FUNCTION: getDatasetResizeConcatenateHorizontal
-----------------------------------------------
Parameters:
	width, height - the dimensions to resize all painting images to

Returns: (train, train_labels, val, val_labels, test, test_labels) where all
non-label tensors are N x HEIGHT x 2*WIDTH x 3 and the label tensors are
N x 1, where N = number of pairs.
-----------------------------------------------
'''
def getDatasetResizeConcatenateHorizontal(width, height):
	dirName = "images-resize-" + width + "x" + height
	datasets = loadDatasetResize(width, height, dirName)
	newDatasets = []

	for dataset in datasets:
		inputs = []
		ys = []

		for entry in dataset:
			ys.append(entry[2])

			image1 = misc.imread(dirName + "/" + entry[0].imageFilename())
			image2 = misc.imread(dirName + "/" + entry[1].imageFilename())
			inputs.append(np.concatenate((image1, image2), axis=1))

		newDatasets.append(np.conatenate(inputs))
		newDatasets.append(np.array(ys))

	return newDatasets

'''
FUNCTION: getDatasetResizeStack
-----------------------------------------------
Parameters:
	width, height - the dimensions to resize all painting images to

Returns: (train, train_labels, val, val_labels, test, test_labels) where all
non-label tensors are N x HEIGHT x WIDTH x 6 and the label tensors are
N x 1, where N = number of pairs.
-----------------------------------------------
'''
def getDatasetResizeStack(width, height):
	dirName = "images-resize-" + width + "x" + height
	datasets = loadDatasetResize(width, height, dirName)
	newDatasets = []

	for dataset in datasets:
		inputs = []
		ys = []

		for entry in dataset:
			ys.append(entry[2])

			image1 = misc.imread(dirName + "/" + entry[0].imageFilename())
			image2 = misc.imread(dirName + "/" + entry[1].imageFilename())
			inputs.append(np.concatenate((image1, image2), axis=2))

		newDatasets.append(np.conatenate(inputs))
		newDatasets.append(np.array(ys))

	return newDatasets






