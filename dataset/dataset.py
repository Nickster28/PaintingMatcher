'''
FILE: dataset.py
----------------
This file contains functions to parse, generate, and preprocess the paintings
theme dataset.  The main functions you will probably want to use are:

* createTrainValTestDatasets: this function reads in the dataset from
dataset.csv and saves to file 3 pickle files: train.pickle, val.pickle, and
test.pickle.  Each pickle file is a list of lists, where each inner list has the
format [PAINTING, PAINTING, SCORE].  each the paintings are Painting objects,
and the score is SAME_THEME (1) or DIFFERENT_THEME (0).  There are 120K pairs in
each pickle file.

* 
----------------
'''


import csv
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool 
from urllib import urlretrieve
from collections import defaultdict
import pickle
import random
import progressbar
import os
import itertools
from PIL import Image
from scipy import misc
import numpy as np

PORTRAIT = 1
NON_PORTRAIT = 0


    #######################################################################
    ########################## UTILITY CLASSES ############################
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
		directory - the directory in which to fix the grayscale structure.
	Returns: NA

	If the image has not already been doctored, check if it has only 1 channel
	(aka grayscale).  If it does, convert it to 3 channels and mark it as
	fixed.
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
	portraitThemes = ['female-portraits', 'male-portraits']
	portraits = []
	for portraitTheme in portraitThemes:
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
Returns: NA


---------------------------------------
'''
def createTrainValTestDatasets():
	# Load the dataset from the pickle file or recreate as a backup
	try:
		dataset = pickle.load(open("downloadedDataset.pickle", "rb"))
	except:
		dataset = generateDataset()
		pickle.dump(dataset, open("downloadedDataset.pickle", "wb"))

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
Parameters: NA
Returns: a (train, val, test) tuple where each entry is a list of
	(painting, score) tuples.  The painting images have not been modified
	other than converting 1-channel grayscale images to 3-channel RGB images.
------------------------
'''
def loadDatasetRaw(numPairs):
	# Load the dataset from the pickle files
	train = pickle.load(open("train.pickle", "rb"), encoding='latin1')
	val = pickle.load(open("val.pickle", "rb"), encoding='latin1')
	test = pickle.load(open("test.pickle", "rb"), encoding='latin1')

	# TODO

