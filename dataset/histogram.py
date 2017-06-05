import tensorflow as tf
import numpy as np

def foo():
	image_string = tf.read_file('images/1.jpg')
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
	image = tf.cast(image_decoded, tf.float32)
	print(tf.Session().run(image).shape)

	# Produce color histogram
	# https://stackoverflow.com/questions/34130902/create-color-histogram-of-an-image-using-tensorflow
	with tf.variable_scope('color_hist_producer') as scope:
		bin_size = 16
		hist_entries = [[], [], []]
		# Split image into single channels
		img_r, img_g, img_b = tf.split(image, 3, 2)
		for channel_i, img_chan in enumerate([img_r, img_g, img_b]):
			for idx, i in enumerate(np.arange(0, 255, bin_size)):
				gt = tf.greater(img_chan, i)
				leq = tf.less_equal(img_chan, i + bin_size)
				# Put together with logical_and, cast to float and sum up entries -> gives count for current bin.
				hist_entries[i], .append(tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32)))

		# Pack scalars together to a tensor, then normalize histogram.
		hist = tf.nn.l2_normalize(tf.stack(hist_entries), 0)

	return tf.Session().run(hist)