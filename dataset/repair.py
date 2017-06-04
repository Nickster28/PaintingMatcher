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
import numpy as np
from dataset import Painting

def fixCorruption(painting):
	print("\n##########################################")
	print("Corrupted painting: " + str(painting.imageFilename()))
	print("Theme: " + painting.theme)
	print("Title: " + painting.title)
	print("Artist: " + painting.artist)
	print("Style: " + painting.style)
	print("Genre: " + painting.genre)
	print("Wiki URL: " + painting.wikiURL)
	print("Image URL: " + painting.imageURL)
	print("##########################################\n")

	painting.grayscaleFixed = False
	try:
		new = raw_input("Theme? ")
		painting.theme = new
	except:
		pass

	try:
		new = raw_input("Title? ")
		painting.title = new
	except:
		pass

	try:
		new = raw_input("Artist? ")
		painting.artist = new
	except:
		pass

	try:
		new = raw_input("Style? ")
		painting.style = new
	except:
		pass

	try:
		new = raw_input("Genre? ")
		painting.genre = new
	except:
		pass

	try:
		new = raw_input("Wiki URL? ")
		painting.wikiURL = new
	except:
		pass

	while True:
		try:
			new = raw_input("Image URL? ")
			painting.imageURL = new
			print("Set Image URL to " + painting.imageURL)
			painting.downloadImageTo("images")
			break
		except Exception as e:
			print(e)

train = pickle.load(open("train.pickle", "rb"))
val = pickle.load(open("val.pickle", "rb"))
test = pickle.load(open("test.pickle", "rb"))

# Create a set of all paintings
allPaintings = set()
for dataset in [train, val, test]:
	for entry in dataset:
		allPaintings.add(entry[0])
		allPaintings.add(entry[1])

# Count the number of corrupted paintings
corruptedPaintings = []
for painting in allPaintings:
	try:
		painting.fixGrayscale("images")
	except Exception as e:
		corruptedPaintings.append(painting)		

print(str(len(corruptedPaintings)) + " corrupted")

# Go through each painting and fix it
for i, painting in enumerate(corruptedPaintings):
	print("Fixing painting " + str(i+1) + " of " + str(len(corruptedPaintings)))
	fixCorruption(painting)
	print("Saving to file")
	pickle.dump(train, open("train.pickle", "wb"))
	pickle.dump(val, open("val.pickle", "wb"))
	pickle.dump(test, open("test.pickle", "wb"))


