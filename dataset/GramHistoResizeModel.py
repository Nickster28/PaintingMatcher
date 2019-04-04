from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
from SimpleResizeModel import SimpleResizeModel
from utils import gram_matrix, histogram
import tensorflow as tf
import numpy as np

class GramHistoResizeModel(SimpleResizeModel):

	def processInputData(self, *args):
		image, label = super(GramHistoResizeModel, self).processInputData(*args)
		return image, histogram(image), label

	def vggInput(self, inputs):
		image = inputs[0]
		histogramTensor = inputs[1]
		
		conv_out = tf.layers.conv2d(image, 32, (7, 7), padding='same', activation=tf.nn.relu)
		gram = gram_matrix(conv_out)
		gram_out = tf.reshape(gram, [-1, 1024])
		gram_out = tf.layers.dense(gram_out, 224, activation=tf.nn.relu)
		gram_out = tf.reshape(gram_out, [-1, 224, 1, 1])

		histogramTensor = tf.reshape(histogramTensor, [-1, 48])
		hist_out = tf.layers.dense(histogramTensor, 224, activation=tf.nn.relu)
		hist_out = tf.reshape(hist_out, [-1, 224, 1, 1])

		return image + gram_out + hist_out