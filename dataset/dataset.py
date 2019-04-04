'''
FILE: dataset.py
----------------
This file contains functions to parse, generate, and preprocess the paintings
theme dataset.  The main functions to use are:

* createTrainValTestDatasets: this function reads in the dataset from
dataset.csv and saves to file 3 pickle files: train.pickle, val.pickle, and
test.pickle.  Each pickle file is a list of lists, where each inner list has the
format [PAINTING, SCORE].  each of the paintings are Painting objects,
and the score is PORTRAIT (1) or NON_PORTRAIT (0).  3/5 of the samples are put
into train, 1/5 into val, and 1/5 into test.

* loadDatasetRaw: this function returns a list of (trainInput, trainLabels,
valInput, valLabels, testInput, testLabels) where each entry is a list of either
Painting objects or ints (PORTRAIT or NON_PORTRAIT).  3/5 of the samples are
put into train, 1/5 into val, and 1/5 into test.
----------------
'''

import csv
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
from urllib.request import urlretrieve
from collections import defaultdict
import pickle
import random
import progressbar
import itertools
import os
import urllib
from scipy import misc
import numpy as np

PORTRAIT = 1
NON_PORTRAIT = 0
PORTRAIT_THEMES = ['female-portraits', 'male-portraits']

    #######################################################################
    ########################### UTILITY CLASS #############################
    #######################################################################

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
		self.grayscaleFixed = False

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

	Returns: True on success, False on failure

	Attempts to download the image for this painting into the given directory if
	it has not been downloaded to this location already (in which case we do
	nothing).  The image will be saved as ID.jpg, where ID is the ID of this
	painting.  If the download fails, the image is deleted.
	---------------------
	'''
	def downloadImageTo(self, directory):
		fullFilename = directory + "/" + self.imageFilename()
		if os.path.isfile(fullFilename):
			return True

		try:
			urlretrieve(self.imageURL, fullFilename)
		except Exception as e:
			if os.path.isfile(fullFilename):
				os.remove(fullFilename)
			return False

		try:
			image = misc.imread(fullFilename)
			return True
		except IOError as e:
			os.remove(fullFilename)
			return False

	'''
	METHOD: deleteImage
	-------------------
	Parameters:
		directory - the directory in which to save this image

	Returns: NA

	Removes the image file for this painting from the given directory.
	-------------------
	'''
	def deleteImage(self, directory):
		fullFilename = directory + "/" + self.imageFilename()
		os.remove(fullFilename)

	'''
	METHOD: imageDownloaded
	-----------------------
	Parameters:
		directory - the directory in which to check for this painting's image.

	Returns: whether or not this painting was successfully downloaded to the
	given directory.
	-----------------------
	'''
	def imageDownloaded(self, directory):
		return os.path.isfile(directory + "/" + self.imageFilename())

	'''
	METHOD: fixGrayscale
	--------------------
	Parameters:
		directory - the directory where this image was previously saved, and
					thus where to go to fix the grayscale structure.
	Returns: NA

	If the image has not already been doctored, check if it has only 1 channel
	(aka grayscale).  If it does, convert it to 3 channels of that same value
	and mark it as fixed.
	--------------------
	'''
	def fixGrayscale(self, directory):
		if self.grayscaleFixed:
			return

		image = misc.imread(directory + "/" + self.imageFilename())
		if len(image.shape) != 3:
		    newImage = np.zeros(image.shape + (3,))
		    for row in range(image.shape[0]):
		        for col in range(image.shape[1]):
		            newImage[row][col][0] = image[row][col]
		            newImage[row][col][1] = image[row][col]
		            newImage[row][col][2] = image[row][col]
		    misc.imsave(directory + "/" + self.imageFilename(), newImage)

		self.grayscaleFixed = True

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
FUNCTION: downloadPaintings
---------------------------
Parameters:
	paintings - the list of painting objects to download images for

Returns: a subset of paintings that have had their images downloaded to the
images/ directory.  Images are downloaded in parallel.  Thanks to
http://chriskiehl.com/article/parallelism-in-one-line/ for guide to parallel
programming in Python.
---------------------------
'''
def downloadPaintings(paintings, batchSize=50):

	def downloadPainting(painting):
		painting.downloadImageTo("images")

	pool = ThreadPool(batchSize)
	pool.map(downloadPainting, paintings)
	pool.close()
	pool.join()

	return [p for p in paintings if p.imageDownloaded("images")]

'''
FUNCTION: generateDataset
------------------------------
Parameters: NA

Returns: a randomized list of portrait and non-portrait paintings, 50% of which
are portraits (male + female) and 50% of which are non-portraits, chosen
equally from all other themes with >= 100 samples.
------------------------------
'''
def generateDataset():
	themesDict = parse('dataset.csv')

	# Remove all themes with few paintings
	MIN_PAINTINGS = 300
	themesDict = {theme: themesDict[theme] for theme in themesDict if len(themesDict[theme]) > MIN_PAINTINGS}
	
	# Gather all 7681 portrait paintings
	portraits = []
	for portraitTheme in PORTRAIT_THEMES:
		portraits += themesDict[portraitTheme]
		del themesDict[portraitTheme]

	print("Downloading portraits...")
	portraits = downloadPaintings(portraits)
	print(str(len(portraits)) + " portraits downloaded")

	minPaintingCount = min([len(themesDict[theme]) for theme in themesDict])

	# Trim all remaining themes to the same size and gather them together
	others = []
	for theme in themesDict:
		paintings = themesDict[theme]
		random.shuffle(paintings)
		others += paintings[:minPaintingCount]
	random.shuffle(others)

	print("Downloading others...")
	others = downloadPaintings(others)
	print(str(len(others)) + " others downloaded")

	# Remove any excess paintings
	numExtras = len(others) - len(portraits)
	for i in range(numExtras):
		p = others[len(portraits) + i]
		p.deleteImage("images")
	others = others[:len(portraits)]

	dataset = portraits + others
	random.shuffle(dataset)
	return dataset
		

'''
MAIN FUNCTION: createTrainValTestDatasets():
---------------------------------------
Parameters: NA
Returns: (train, val, test) tuple of lists where each list contains tuples of
the format (PAINTING, SCORE).

Labels the downloaded dataset and splits it into 2/5 train, 2/5 val, and 1/5
test datasets.  Each dataset is a list of tuples where the first entry is a
painting and the second entry is the score (PORTRAIT or NON_PORTRAIT).  The
painting images have not been modified other than converting 1-channel grayscale
images to 3-channel RGB images.
---------------------------------------
'''
def createTrainValTestDatasets():
	# Load the dataset from the pickle file or recreate as a backup
	try:
		dataset = pickle.load(open("downloadedDataset.pickle", "rb"), encoding='latin1')
	except IOError as e:
		dataset = generateDataset()
		pickle.dump(dataset, open("downloadedDataset.pickle", "wb"))

	# Fix grayscale
	bar = progressbar.ProgressBar()
	for i in bar(range(len(dataset))):
		dataset[i].fixGrayscale("images")

	labeledDataset = [(p, PORTRAIT if p.theme in PORTRAIT_THEMES else NON_PORTRAIT) for p in dataset]
		
	# Split the pairs into train, val and test
	numPerSubset = int(len(dataset) / 5)

	train = labeledDataset[: 3 * numPerSubset]
	val = labeledDataset[3 * numPerSubset : 4 * numPerSubset]
	test = labeledDataset[4 * numPerSubset :]

	return (train, val, test)


    #######################################################################
    ########################## DATASET LOADING ############################
    #######################################################################


'''
FUNCTION: loadDatasetRaw
------------------------
Parameters:
	size - optionally specify the size of the three datasets

Returns: a (trainInput, trainLabels, valInput, valLabels, testInput, testLabels)
list where each entry is either a list of Painting objects (input) or a list of
scores (label).
------------------------
'''
def loadDatasetRaw(size=-1):
	# Load the datasets from the pickle file or recreate as a backup
	try:
		train = pickle.load(open("train.pickle", "rb"), encoding='latin1')
		val = pickle.load(open("val.pickle", "rb"), encoding='latin1')
		test = pickle.load(open("test.pickle", "rb"), encoding='latin1')
	except IOError as e:
		(train, val, test) = createTrainValTestDatasets()
		pickle.dump(train, open("train.pickle", "wb"))
		pickle.dump(val, open("val.pickle", "wb"))
		pickle.dump(test, open("test.pickle", "wb"))

	trainInput = [entry[0] for entry in train]
	trainLabels = [entry[1] for entry in train]
	valInput = [entry[0] for entry in val]
	valLabels = [entry[1] for entry in val]
	testInput = [entry[0] for entry in test]
	testLabels = [entry[1] for entry in test]

	output = (trainInput, trainLabels, valInput, valLabels, testInput, testLabels)

	# Cap dataset size if needed
	if size > 0:
		output = [o[:size] for o in output]

	return output

