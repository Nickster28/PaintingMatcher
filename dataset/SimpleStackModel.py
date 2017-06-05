from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf

class SimpleStackModel(PaintingThemeModel):

	def getDataset(self):
		(
		    train_pairs,
		    train_labels, 
		    val_pairs, 
		    val_labels, 
		    test_pairs, 
		    test_labels
		) = loadDatasetRaw(20)

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
		for paintingFilename in [filename1, filename2]:
		    image_string = tf.read_file(paintingFilename)
		    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
		    image = tf.cast(image_decoded, tf.float32)

		    resized_image = tf.image.resize_images(image, [224, 224])  # (2)
		    resized_images.append(resized_image)

		return tf.concat(resized_images, 2), label

	def vggInput(self, inputs):
		# Added: an additional layer taking our input tensors and reshaping them
		W_pre = tf.get_variable("W_pre", shape=[7,7,6,3])
		b_pre = tf.get_variable("b_pre", shape=[3])
		pre_out = tf.nn.conv2d(inputs[0], W_pre, strides=[1,1,1,1], padding='SAME') + b_pre
		return tf.nn.relu(pre_out)

model = SimpleStackModel()
model.train()
