from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf
import numpy as np

lexicon = cPickle.load(open('../data/glove/trimmed_word_lexicon.p', 'r')) 
titles_file = np.load('image_titles.npy')
embedding = np.load('titles_trimmed_glove_vectors.npy')

class TitlesStackModel(PaintingThemeModel):
            
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
        title_embeddings = []
        for paintingFilename in [filename1, filename2]:
            image_string = tf.read_file(paintingFilename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            resized_image = tf.image.resize_images(image, [224, 224])  # (2)
            resized_images.append(resized_image)

            img_title = titles_file[paintingFilename]
            lower_title = []
            counter = 0
            for word in img_title:
                if counter >= 5:
                    break
                if counter >= len(img_title):
                    lower_title.append(lexicon['<UNK>'])
                else:
                    try:
                        lower_title.append(lexicon[word.lower()])
                    except:
                        lower_title.append(lexicon['<UNK>'])
                counter += 1
            
            W = tf.get_variable(name='W', shape=embedding.shape, tf.constant_initializer(embedding), trainable=False)
            title_embed = tf.nn.lookup(W, tf.convert_to_tensor(lower_title))
            title_embeddings.append(title_embed)

        return tf.concat(resized_images, 2), tf.concat(title_embeddings, 0), label

    def vggInput(self, inputs):
        imageTensor = inputs[0]
        histogramTensor = inputs[1]

        # Added: an additional layer taking our input tensors and reshaping them
        conv_out = tf.layers.conv2d(inputs[0], 3, (7, 7), padding='same', activation=tf.nn.relu)
        #W_hist = tf.get_variable("W_hist", shape=[X, 224*224*3])
        #b_hist = tf.get_variable("b_hist", shape=[224*224*3])
        #dense_out = tf.matmul(histogramTensor, W_hist) + b_hist
        return conv_out #+ dense_out

model = TitlesStackModel()
model.train()
