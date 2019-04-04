import tensorflow as tf
import numpy as np

def histogram(image):
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
		return histogram

def gram_matrix(features, normalize=True):
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