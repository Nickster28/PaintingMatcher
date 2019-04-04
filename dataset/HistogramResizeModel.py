from SimpleResizeModel import SimpleResizeModel
from utils import histogram
import tensorflow as tf

class HistogramResizeModel(SimpleResizeModel):

	def processInputData(self, *args):
		image, label = super(HistogramResizeModel, self).processInputData(*args)
		return image, histogram(image), label

	def vggInput(self, inputs):
		imageTensor = inputs[0]
		histogramTensor = inputs[1]

		# Added: an additional layer taking our input tensors and reshaping them
		histogramTensor = tf.reshape(histogramTensor, [-1, 48])
		hist_out = tf.layers.dense(histogramTensor, 224, activation=tf.nn.relu)
		hist_out = tf.reshape(hist_out, [-1, 224, 1, 1])
		return imageTensor + hist_out