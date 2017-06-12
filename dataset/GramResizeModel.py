from SimpleResizeModel import SimpleResizeModel
import tensorflow as tf
from utils import gram_matrix

class GramResizeModel(SimpleResizeModel):

	'''
	METHOD: gramLayer
	-----------------
	Parameters:
		image - the input image (N x 224 x 224 x 3)

	Returns: the output gram tensor for that image.  It is created by passing
	the image through a conv layer with 32 7x7 filters and then calculating the
	gram matrix of that output which is reshaped to N x 224 x 1 x 1.
	-----------------
	'''
	def gramLayer(self, image):
		conv_out = tf.layers.conv2d(image, 32, (7, 7), padding='same', activation=tf.nn.relu)
		gramTensor = gram_matrix(conv_out)

		# Added: an additional layer taking our input tensors and reshaping them
		gram_out = tf.reshape(gramTensor, [-1, 1024])
		gram_out = tf.layers.dense(gram_out, 224, activation=tf.nn.relu)
		return tf.reshape(gram_out, [-1, 224, 1, 1])

	def vggInput(self, inputs):
		imageTensor = inputs[0]
		gramTensor = self.gramLayer(imageTensor)
		return imageTensor + gramTensor