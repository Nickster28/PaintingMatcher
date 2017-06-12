from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf
import numpy as np

class HistogramStackModel(PaintingThemeModel):

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
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
		image = tf.cast(image_decoded, tf.float32)

		resized_image = tf.image.resize_images(image, [224, 224])  # (2)

		# Produce color histogram
		# https://stackoverflow.com/questions/34130902/create-color-histogram-of-an-image-using-tensorflow
		with tf.variable_scope('color_hist_producer') as scope:
			bin_size = 16
			hist_entries = []
			# Split image into single channels
			for img_chan in tf.split(image, 3, 2):
				hist_entries.append([])
				for bin_val in np.arange(0, 255, bin_size):
					gt = tf.greater(img_chan, bin_val)
					leq = tf.less_equal(img_chan, bin_val + bin_size)
					# Put together with logical_and, cast to float and sum up entries -> gives count for current bin.
					hist_entries[-1].append(tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32)))

			# Pack scalars together to a tensor, then normalize histogram.
			hist_entries = [row / sum(row) for row in hist_entries]
			histogram = tf.stack(hist_entries)			

		return resized_image, histogram, label

	def vggInput(self, inputs):
		imageTensor = inputs[0]
		histogramTensor = inputs[1]

		# Added: an additional layer taking our input tensors and reshaping them
		conv_out = tf.layers.conv2d(imageTensor, 3, (7, 7), padding='same', activation=tf.nn.relu)
		histogramTensor = tf.reshape(histogramTensor, [-1, 96])
		hist_out = tf.layers.dense(histogramTensor, 224, activation=tf.nn.relu)
		hist_out = tf.reshape(hist_out, [-1, 224, 1, 1])
		return conv_out + hist_out

model = HistogramStackModel()
model.train()