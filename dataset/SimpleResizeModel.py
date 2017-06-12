from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf

class SimpleResizeModel(PaintingThemeModel):

	# Resizes the image to 224x224 using tf.image.resize_images
	def processInputData(self, *args):
		filename = args[0]
		label = args[1]

		# Read in image
		image_string = tf.read_file(filename)
		image_decoded = tf.image.decode_jpeg(image_string, channels=3)
		image = tf.cast(image_decoded, tf.float32) # H x W

		return tf.image.resize_images(image, [224, 224]), label