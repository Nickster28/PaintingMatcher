import os
import urllib
from scipy import misc
import numpy as np

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

	Returns: NA

	Attempts to download the image for this painting into the given directory if
	it has not been downloaded to this location already (in which case we do
	nothing).  The image will be saved using this painting's filename.
	If the download fails, this throws an exception.
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