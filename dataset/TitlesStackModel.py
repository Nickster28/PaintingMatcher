from model import PaintingThemeModel, Painting
from dataset import loadDatasetRaw
import tensorflow as tf
import numpy as np
import _pickle as cPickle

lexicon = cPickle.load(open('trimmed_word_lexicon.p', 'rb')) 
embedding = np.load('titles_trimmed_glove_vectors.npy')

class TitlesStackModel(PaintingThemeModel):

    def get_lexicon_indices(self, title_list):
        output = np.zeros((self.dataset_size, 5, 300))
        for i, img_title in enumerate(title_list):

            # Add at most 5 words from img_title
            for j in range(0, min(len(img_title), 5)):
                vectorIndex = lexicon.get(img_title[j].lower(), lexicon['<UNK>'])
                output[i,j,:] = embedding[vectorIndex]

            # Add padding up to 5 words
            for j in range(0, max(5 - len(img_title), 0)):
                output[i,j,:] = embedding[lexicon['<UNK>']]

        return output
            
    def getDataset(self):
        (
            train_pairs,
            train_labels, 
            val_pairs, 
            val_labels, 
            test_pairs, 
            test_labels
        ) = loadDatasetRaw(self.dataset_size)
        
        titles_file = np.load('image_titles.npy').item()
        print(titles_file['158.jpg'])
        

        train_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), train_pairs))
        train_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), train_pairs))
        train_pairs_1_tmp = list(map(lambda pair: pair[0].imageFilename(), train_pairs))
        train_pairs_2_tmp = list(map(lambda pair: pair[1].imageFilename(), train_pairs))
        train_pairs_1_titles = self.get_lexicon_indices([titles_file[title] for title in train_pairs_1_tmp])
        train_pairs_2_titles = self.get_lexicon_indices([titles_file[title] for title in train_pairs_2_tmp])
        train_pairs_1 = tf.constant(train_pairs_1)
        train_pairs_2 = tf.constant(train_pairs_2)
        train_pairs_1_titles = tf.constant(train_pairs_1_titles, dtype=tf.float32)
        train_pairs_2_titles = tf.constant(train_pairs_2_titles, dtype=tf.float32)
        

        val_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), val_pairs))
        val_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), val_pairs))
        val_pairs_1_tmp = list(map(lambda pair: pair[0].imageFilename(), val_pairs))
        val_pairs_2_tmp = list(map(lambda pair: pair[1].imageFilename(), val_pairs))
        val_pairs_1_titles = self.get_lexicon_indices([titles_file[title] for title in val_pairs_1_tmp])
        val_pairs_2_titles = self.get_lexicon_indices([titles_file[title] for title in val_pairs_2_tmp])
        val_pairs_1 = tf.constant(val_pairs_1)
        val_pairs_2 = tf.constant(val_pairs_2)
        val_pairs_1_titles = tf.constant(val_pairs_1_titles, dtype=tf.float32)
        val_pairs_2_titles = tf.constant(val_pairs_2_titles, dtype=tf.float32)

        return {
            "train": (train_pairs_1, train_pairs_2, train_pairs_1_titles, train_pairs_2_titles, tf.constant(train_labels)),
            "val": (val_pairs_1, val_pairs_2, val_pairs_1_titles, val_pairs_2_titles, tf.constant(val_labels))
        }

    def processInputData(self, *args):
        filename1 = args[0]
        filename2 = args[1]
        label = args[4]

        resized_images = []
        title_embeddings = []
        for paintingFilename in [filename1, filename2]:
            image_string = tf.read_file(paintingFilename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
            image = tf.cast(image_decoded, tf.float32)

            resized_image = tf.image.resize_images(image, [224, 224])  # (2)
            resized_images.append(resized_image)


        return tf.concat(values=resized_images, axis=2), tf.stack(values=args[2:4]), label

    def vggInput(self, inputs):
        imageTensor = inputs[0]
        titlesTensor = inputs[1]

        # Added: an additional layer taking our input tensors and reshaping them
        conv_out = tf.layers.conv2d(inputs[0], 3, (7, 7), padding='same', activation=tf.nn.relu)
        titlesTensor = tf.reshape(titlesTensor, [-1, 3000])
        title_out = tf.layers.dense(titlesTensor, 224, activation=tf.nn.relu)
        title_out = tf.reshape(title_out, [-1, 224, 1, 1])
        return conv_out + title_out

model = TitlesStackModel()
model.train()
