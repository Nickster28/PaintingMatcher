from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf

class SimpleStackModel(PaintingThemeModel):

	def getDataset(self, size=-1):
		(
		    trainInput,
		    trainLabels, 
		    valInput, 
		    valLabels, 
		    testInput, 
		    testLabels
		) = loadDatasetRaw(size=size)

		trainFilenames = list(map(lambda p: "images/" + p.imageFilename(), trainInput))
		valFilenames = list(map(lambda p: "images/" + p.imageFilename(), valInput))
		testFilenames = list(map(lambda p: "images/" + p.imageFilename(), testInput))
		testThemes = list(map(lambda p: p.theme, testInput))

		return {
			"train": (tf.constant(trainFilenames), tf.constant(trainLabels)),
			"val": (tf.constant(valFilenames), tf.constant(valLabels)),
			"test": (tf.constant(testFilenames), tf.constant(testLabels))
		}

	def processInputData(self, *args):
		filename = args[0]
		label = args[1]

		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)  # (1)
		image = tf.cast(image_decoded, tf.float32)

		resized_image = tf.image.resize_images(image, [224, 224])  # (2)		    

		return resized_image, label

model = SimpleStackModel()
model.train()
