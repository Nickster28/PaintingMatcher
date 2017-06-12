"""
VGG-based model for classifying paintings based on whether they are
thematically "portraits" or not.  Relies on the VGGNet architecture as a
base, plus custom additional layers at the beginning.

Uses tf.contrib.data module which is in release candidate 1.2.0rc0
Based on:
    - PyTorch example from Justin Johnson:
      https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
Required packages: tensorflow (v1.2)
You can install the release candidate 1.2.0rc0 here:
https://www.tensorflow.org/versions/r1.2/install/
"""

import argparse
import shutil
import pickle

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import matplotlib.pyplot as plt
import numpy as np

from dataset import *

class PaintingThemeModel:
    NUM_CLASSES = 2

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--batch_size', default=32, type=int)
        self.parser.add_argument('--dataset_size', default=-1, type=int)
        self.parser.add_argument('--num_workers', default=4, type=int)
        self.parser.add_argument('--num_epochs', default=10, type=int)
        self.parser.add_argument('--learning_rate', default=0.01, type=float)
        self.parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
        self.parser.add_argument('--weight_decay', default=5e-4, type=float)
        self.parser.add_argument('--log_dir', default='log', type=str)

    # Should return the ultimate input row given a row from each inputted
    # column from our dataset.
    def processInputData(self, *args):
        raise NotImplementedError

    # Should return a train and val element, each containing all necessary
    # components to go into the Dataset object.
    def getDataset(self, size=-1):
        raise NotImplementedError

    # Should add any additional graph components and return the input to VGG.
    # Defaults to just returning the first input value.
    def vggInput(self, inputs):
        return inputs[0]

    def check_accuracy(self, sess, correct_prediction, is_training,
        dataset_init_op):
        """
        Check the accuracy of the model on either train or val
        (depending on dataset_init_op).
        """

        # Initialize the correct dataset
        sess.run(dataset_init_op)
        num_correct, num_samples = 0, 0
        full_correct = []
        while True:
            try:
                correct_pred = sess.run(correct_prediction, {
                    is_training: False
                })

                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0]
                full_correct.append(correct_pred)
            except tf.errors.OutOfRangeError:
                break


        # Return the fraction of datapoints that were correctly classified
        acc = float(num_correct) / num_samples
        return (acc, np.concatenate(full_correct))

    def train(self):
        args = self.parser.parse_args()
        
        """
        ------------------------------------------------------------------------
        In TensorFlow, you first want to define the computation graph with all
        the necessary operations: loss, training op, accuracy...
        Any tensors created in the `graph.as_default()` scope will be part of
        `graph`
        ------------------------------------------------------------------------
        """
        graph = tf.Graph()
        with graph.as_default():
            """
            Derived from standard preprocessing for VGG on ImageNet from here:
            https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
            and here: https://arxiv.org/pdf/1409.1556.pdf

            --------------------------------------------------------------------
            DATASET CREATION using tf.contrib.data.Dataset
            https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

            The tf.contrib.data.Dataset framework uses queues in the
            background to feed in data to the model.
            We initialize the dataset with a list of filenames and labels, and
            then apply the preprocessing functions described above.
            Behind the scenes, queues will load the filenames, preprocess them
            with multiple threads and apply the preprocessing in parallel, and
            then batch the data.
            --------------------------------------------------------------------
            """
            dataset = self.getDataset(size=args.dataset_size)

            # Training dataset
            train_dataset = tf.contrib.data.Dataset.from_tensor_slices(dataset["train"])
            train_dataset = train_dataset.map(self.processInputData,
                num_threads=args.num_workers, output_buffer_size=args.batch_size)
            # don't forget to shuffle
            train_dataset = train_dataset.shuffle(buffer_size=10000)
            batched_train_dataset = train_dataset.batch(args.batch_size)

            # Validation dataset
            val_dataset = tf.contrib.data.Dataset.from_tensor_slices(dataset["val"])
            val_dataset = val_dataset.map(self.processInputData,
                num_threads=args.num_workers, output_buffer_size=args.batch_size)
            batched_val_dataset = val_dataset.batch(args.batch_size)

            # Test dataset
            test_dataset = tf.contrib.data.Dataset.from_tensor_slices(dataset["test"])
            test_dataset = test_dataset.map(self.processInputData,
                num_threads=args.num_workers, output_buffer_size=args.batch_size)
            batched_test_dataset = test_dataset.batch(args.batch_size)

            """
            --------------------------------------------------------------------
            Now we define an iterator that can operator on either dataset.
            The iterator can be reinitialized by calling:
                - sess.run(train_init_op) for 1 epoch on the training set
                - sess.run(val_init_op)   for 1 epoch on the valiation set
            Once this is done, we don't need to feed any value for images and
            labels as they are automatically pulled out from the iterator
            queues.

            A reinitializable iterator is defined by its structure. We could
            use the `output_types` and `output_shapes` properties of either
            `train_dataset` or `validation_dataset` here, because they are
            compatible.
            --------------------------------------------------------------------
            """
            iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                               batched_train_dataset.output_shapes)
            iterator_output = iterator.get_next()
            model_inputs = iterator_output[:-1]
            labels = iterator_output[-1]

            train_init_op = iterator.make_initializer(batched_train_dataset)
            val_init_op = iterator.make_initializer(batched_val_dataset)
            test_init_op = iterator.make_initializer(batched_test_dataset)

            # Indicates whether we are in training or in test mode
            is_training = tf.placeholder(tf.bool)

            """
            --------------------------------------------------------------------
            Now that we have set up the data, it's time to set up the model.
            We use the VGG-16 model architecture from the ImageNet challenge as
            a base, and add additional layers at the front. We also remove the
            last fully connected layer (fc8) and replace it with our own, with
            an output size num_classes=2.

            Get the VGG model, specifying the num_classes argument to create a
            new fully connected layer replacing the last one, called
            "vgg_16/fc8".  Each model has a different architecture, so
            "vgg_16/fc8" will change in another model.  Here, logits gives us
            directly the predicted scores we wanted from the images.
            --------------------------------------------------------------------
            """
            vgg_input = self.vggInput(model_inputs)
            vgg = tf.contrib.slim.nets.vgg
            with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
                logits, _ = vgg.vgg_16(vgg_input, num_classes=PaintingThemeModel.NUM_CLASSES,
                                        is_training=is_training,
                                        dropout_keep_prob=args.dropout_keep_prob)

            """
            --------------------------------------------------------------------
            Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES
            collection We can then call the total loss easily.
            --------------------------------------------------------------------
            """
            tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            loss = tf.losses.get_total_loss()
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
            train_op = optimizer.minimize(loss)

            # Evaluation metrics
            prediction = tf.to_int32(tf.argmax(logits, 1))
            correct_prediction = tf.equal(prediction, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        """
        ------------------------------------------------------------------------
        Now that we have built the graph, we define the session.
        The session is the interface to *run* the computational graph.
        E.g. we can call our training operations with `sess.run(train_op)'
        ------------------------------------------------------------------------
        """
        with tf.Session(graph=graph) as sess:

            # Tensorboard logging
            train_writer = tf.summary.FileWriter(args.log_dir + '/train', graph)
            merged_summary = tf.summary.merge_all()

            sess.run(tf.global_variables_initializer())
            for epoch in range(args.num_epochs):
                print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs))
                sess.run(train_init_op)
                i = 0
                while True:
                    try:
                        summary, _ = sess.run([merged_summary, train_op], {is_training: True})
                        train_writer.add_summary(summary, epoch * int(train_dataset.shape[0]) / args.batch_size + i)
                        i += 1
                    except tf.errors.OutOfRangeError:
                        break

                # Check accuracy on the train and val sets every epoch
                train_acc, _ = self.check_accuracy(sess, correct_prediction, is_training, train_init_op)
                val_acc, _ = self.check_accuracy(sess, correct_prediction, is_training, val_init_op)
                print('Train accuracy: %f' % train_acc)
                print('Val accuracy: %f\n' % val_acc)

            # Check accuracy on the test set at the end
            test_acc, answers = self.check_accuracy(sess, correct_prediction, is_training, test_init_op)
            print('Test accuracy: %f' % test_acc)
            np.save("test_correct", answers)

            train_writer.close()

