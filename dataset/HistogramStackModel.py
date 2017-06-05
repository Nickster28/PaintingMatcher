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
		) = loadDatasetRaw(200)

		train_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), train_pairs))
		train_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), train_pairs))
		train_pairs_1 = tf.constant(train_pairs_1)
		train_pairs_2 = tf.constant(train_pairs_2)

		val_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), val_pairs))
		val_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), val_pairs))
		val_pairs_1 = tf.constant(val_pairs_1)
		val_pairs_2 = tf.constant(val_pairs_2)

		return {
			"train": (train_pairs_1, train_pairs_2, tf.constant(train_labels)),
			"val": (val_pairs_1, val_pairs_2, tf.constant(val_labels))
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
		        bin_size = 0.2
		        hist_entries = []
		        # Split image into single channels
		        img_r, img_g, img_b = tf.split(resized_image, 3, 2)
		        for img_chan in [img_r, img_g, img_b]:
		            for idx, i in enumerate(np.arange(-1, 1, bin_size)):
		                gt = tf.greater(img_chan, i)
		                leq = tf.less_equal(img_chan, i + bin_size)
		                # Put together with logical_and, cast to float and sum up entries -> gives count for current bin.
		                hist_entries.append(tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32)))

		        # Pack scalars together to a tensor, then normalize histogram.
		        hist = tf.nn.l2_normalize(tf.stack(hist_entries), 0)
		        histograms.append(hist)

		hist_diff = histograms[0] - histograms[1]

		return tf.concat(resized_images, 2), hist_diff, label

	def vggInput(self, inputs):
		imageTensor = inputs[0]
		histogramTensor = inputs[1]

		# Added: an additional layer taking our input tensors and reshaping them
		conv_out = tf.layers.conv2d(inputs[0], 3, (7, 7), padding='same', activation=tf.nn.relu)
		W_hist = tf.get_variable("W_hist", shape=[X, 224*224*3])
		b_hist = tf.get_variable("b_hist", shape=[224*224*3])
		dense_out = tf.matmul(histogramTensor, W_hist) + b_hist
		return conv_out + dense_out

model = HistogramStackModel()
model.train()