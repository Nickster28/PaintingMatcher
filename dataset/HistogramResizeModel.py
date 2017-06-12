from model import PaintingThemeModel, Painting
from SimpleResizeModel import SimpleResizeModel
from dataset import loadDatasetRaw
import tensorflow as tf
import numpy as np

class HistogramResizeModel(SimpleResizeModel):

	def processInputData(self, *args):
		image, label = super(HistogramResizeModel, self).processInputData(*args)

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
		histogramTensor = tf.reshape(histogramTensor, [-1, 48])
		hist_out = tf.layers.dense(histogramTensor, 224, activation=tf.nn.relu)
		hist_out = tf.reshape(hist_out, [-1, 224, 1, 1])
		return imageTensor + hist_out