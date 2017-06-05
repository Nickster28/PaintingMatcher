from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf
import numpy as np
import _pickle as cPickle

lexicon = cPickle.load(open('trimmed_word_lexicon.p', 'rb')) 
embedding = np.load('titles_trimmed_glove_vectors.npy')

class TitlesStackModel(PaintingThemeModel):

    def get_lexicon_indices(self, title_list):
        new_title_list = []
        for img_title in title_list:
            lower_title = []
            for i in range(0, 5):
                try:
                    lower_title.append(embedding[lexicon[img_title[i].lower()]])
                except:
                    lower_title.append(embedding[lexicon['<UNK>']])
            new_title_list.append(lower_title)
        return new_title_list
            
    def getDataset(self):
        (
            train_pairs,
            train_labels, 
            val_pairs, 
            val_labels, 
            test_pairs, 
            test_labels
        ) = loadDatasetRaw(200)
        
        titles_file = np.load('image_titles.npy').item()
        

        train_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), train_pairs))
        train_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), train_pairs))
        train_pairs_1_tmp = list(map(lambda pair: pair[0].imageFilename(), train_pairs))
        train_pairs_2_tmp = list(map(lambda pair: pair[1].imageFilename(), train_pairs))
        train_pairs_1_titles = self.get_lexicon_indices([titles_file[title] for title in train_pairs_1_tmp])
        train_pairs_2_titles = self.get_lexicon_indices([titles_file[title] for title in train_pairs_2_tmp])
        train_pairs_1 = tf.constant(train_pairs_1)
        train_pairs_2 = tf.constant(train_pairs_2)
        train_pairs_1_titles = tf.constant(train_pairs_1_titles)
        train_pairs_2_titles = tf.constant(train_pairs_2_titles)
        

        val_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), val_pairs))
        val_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), val_pairs))
        val_pairs_1_tmp = list(map(lambda pair: pair[0].imageFilename(), val_pairs))
        val_pairs_2_tmp = list(map(lambda pair: pair[1].imageFilename(), val_pairs))
        val_pairs_1_titles = self.get_lexicon_indices([titles_file[title] for title in val_pairs_1_tmp])
        val_pairs_2_titles = self.get_lexicon_indices([titles_file[title] for title in val_pairs_2_tmp])
        val_pairs_1 = tf.constant(val_pairs_1)
        val_pairs_2 = tf.constant(val_pairs_2)
        val_pairs_1_titles = tf.constant(val_pairs_1_titles)
        val_pairs_2_titles = tf.constant(val_pairs_2_titles)

        return {
            "train": (train_pairs_1, train_pairs_2, train_pairs_1_titles, train_pairs_2_titles, tf.constant(train_labels)),
            "val": (val_pairs_1, val_pairs_2, val_pairs_1_titles, val_pairs_2_titles, tf.constant(val_labels))
        }

    def processInputData(self, *args):
        filename1 = args[0]
        filename2 = args[1]
        title1 = args[2]
        title2 = args[3]
        label = args[4]

        resized_images = []
        title_embeddings = []
        for paintingFilename in [filename1, filename2]:
            image_string = tf.read_file(paintingFilename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            resized_image = tf.image.resize_images(image, [224, 224])  # (2)
            resized_images.append(resized_image)

        
        for title in [title1, title2]:
            title_embeddings.append(title)
           
        #    W = tf.get_variable(name='W', shape=embedding.shape, initializer=tf.constant_initializer(tf.constant(embedding, dtype='float32')), trainable=False)
        #    title_embed = tf.nn.lookup(W, title)
        #    title_embeddings.append(title_embed)

        return tf.concat(values=resized_images, axis=2), label #tf.stack(values=title_embeddings, axis=0), label

    def vggInput(self, inputs):
        imageTensor = inputs[0]
        titlesTensor = inputs[1]

        # Added: an additional layer taking our input tensors and reshaping them
        conv_out = tf.layers.conv2d(inputs[0], 3, (7, 7), padding='same', activation=tf.nn.relu)
        #W_hist = tf.get_variable("W_hist", shape=[X, 224*224*3])
        #b_hist = tf.get_variable("b_hist", shape=[224*224*3])
        #dense_out = tf.matmul(histogramTensor, W_hist) + b_hist
        return conv_out #+ dense_out

model = TitlesStackModel()
model.train()
