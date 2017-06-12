from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
from SimpleResizeModel import SimpleResizeModel
import tensorflow as tf
import numpy as np

class GramResizeModel(SimpleResizeModel):

	def gram_matrix(self, features, normalize=True):
		"""
		Compute the Gram matrix from features.
		
		Inputs:
		- features: Tensor of shape (1, H, W, C) giving features for
		  a single image.
		- normalize: optional, whether to normalize the Gram matrix
			If True, divide the Gram matrix by the number of neurons (H * W * C)
		
		Returns:
		- gram: Tensor of shape (C, C) giving the (optionally normalized)
		  Gram matrices for the input image.
		"""
		shape = tf.shape(features)
		features = tf.transpose(features, perm=[0, 3, 1, 2])
		features = tf.reshape(features, [shape[0], shape[3], -1])
		gram = tf.matmul(features, tf.transpose(features, perm=[0, 2, 1]))
		if normalize:
			gram /= tf.to_float(shape[1] * shape[2] * shape[3])
		return gram

	def vggInput(self, inputs):
		imageTensor = inputs[0]

		conv_out = tf.layers.conv2d(imageTensor, 32, (7, 7), padding='same', activation=tf.nn.relu)
		gramTensor = self.gram_matrix(conv_out)

		# Added: an additional layer taking our input tensors and reshaping them
		gram_out = tf.reshape(gramTensor, [-1, 1024])
		gram_out = tf.layers.dense(gram_out, 224, activation=tf.nn.relu)
		gram_out = tf.reshape(gram_out, [-1, 224, 1, 1])
		return imageTensor + gram_out