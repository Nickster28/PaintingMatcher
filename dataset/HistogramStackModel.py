from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf
import numpy as np

class HistogramStackModel(PaintingThemeModel):

	def getDataset(self):
		(
		    train_pairs,
		    train_labels, 
		    val_pairs, 
		    val_labels, 
		    test_pairs, 
		    test_labels
		) = loadDatasetRaw(self.dataset_size)

		train_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), train_pairs))
		train_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), train_pairs))
		train_pairs_1 = tf.constant(train_pairs_1)
		train_pairs_2 = tf.constant(train_pairs_2)

		val_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), val_pairs))
		val_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), val_pairs))
		val_pairs_1 = tf.constant(val_pairs_1)
		val_pairs_2 = tf.constant(val_pairs_2)

		test_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), test_pairs))
		test_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), test_pairs))
		test_pairs_1 = tf.constant(test_pairs_1)
		test_pairs_2 = tf.constant(test_pairs_2)

		return {
			"train": (train_pairs_1, train_pairs_2, tf.constant(train_labels)),
			"val": (val_pairs_1, val_pairs_2, tf.constant(val_labels)),
			"test": (test_pairs_1, test_pairs_2, tf.constant(test_labels))
		}

	def processInputData(self, *args):
		filename1 = args[0]
		filename2 = args[1]
		label = args[2]

		resized_images = []
		histograms = []
		for paintingFilename in [filename1, filename2]:
			image_string = tf.read_file(paintingFilename)
			image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
			image = tf.cast(image_decoded, tf.float32)

			resized_image = tf.image.resize_images(image, [224, 224])  # (2)
			resized_images.append(resized_image)

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
				histograms.append(tf.stack(hist_entries))

		return tf.concat(resized_images, 2), tf.stack(histograms), label

	def vggInput(self, inputs):
		imageTensor = inputs[0]
		histogramTensor = inputs[1]

		# Added: an additional layer taking our input tensors and reshaping them
		conv_out = tf.layers.conv2d(inputs[0], 3, (7, 7), padding='same', activation=tf.nn.relu)
		histogramTensor = tf.reshape(histogramTensor, [-1, 96])
		hist_out = tf.layers.dense(histogramTensor, 224, activation=tf.nn.relu)
		hist_out = tf.reshape(hist_out, [-1, 224, 1, 1])
		return conv_out + hist_out

model = HistogramStackModel()
model.train()