"""
Example TensorFlow script for finetuning a VGG model on your own data.
Uses tf.contrib.data module which is in release candidate 1.2.0rc0
Based on:
    - PyTorch example from Justin Johnson:
      https://gist.github.com/jcjohnson/6e41e8512c17eae5da50aebef3378a4c
Required packages: tensorflow (v1.2)
You can install the release candidate 1.2.0rc0 here:
https://www.tensorflow.org/versions/r1.2/install/
Download the weights trained on ImageNet for VGG:
```
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```
"""

import argparse
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets

from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--learning_rate', default=1e-5, type=float)
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


def check_accuracy(sess, correct_prediction, is_training, dataset_init_op):
    """
    Check the accuracy of the model on either train or val (depending on dataset_init_op).
    """
    # Initialize the correct dataset
    sess.run(dataset_init_op)
    num_correct, num_samples = 0, 0
    while True:
        try:
            correct_pred = sess.run(correct_prediction, {is_training: False})
            num_correct += correct_pred.sum()
            num_samples += correct_pred.shape[0]
        except tf.errors.OutOfRangeError:
            break

    # Return the fraction of datapoints that were correctly classified
    acc = float(num_correct) / num_samples
    return acc


def main(args):
    # Get the list of filenames and corresponding list of labels for training et validation
    train_pairs, train_labels, val_pairs, val_labels, test_pairs, test_labels = loadDatasetRaw()
    
    num_classes = 2

    # --------------------------------------------------------------------------
    # In TensorFlow, you first want to define the computation graph with all the
    # necessary operations: loss, training op, accuracy...
    # Any tensor created in the `graph.as_default()` scope will be part of `graph`
    graph = tf.Graph()
    with graph.as_default():
        # Standard preprocessing for VGG on ImageNet taken from here:
        # https://github.com/tensorflow/models/blob/master/slim/preprocessing/vgg_preprocessing.py
        # Also see the VGG paper for more details: https://arxiv.org/pdf/1409.1556.pdf

        # Preprocessing (for both training and validation):
        # (1) Decode the image from jpg format
        # (2) Resize the image
        def _parse_function(filename1, filename2, label):
            resized_images = []
            histograms = []
            for paintingFilename in [filename1, filename2]:
                image_string = tf.read_file(paintingFilename)
                image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
                image = tf.cast(image_decoded, tf.float32)

                resized_image = tf.image.resize_images(image, [224, 224])  # (2)
                resized_images.append(resized_image)

                # Produce color histogram
                with tf.variable_scope('color_hist_producer') as scope:
                    bin_size = 0.2
                    hist_entries = []
                    # Split image into single channels
                    img_r, img_g, img_b = tf.split(centered_image, 3, 2)
                    for img_chan in [img_r, img_g, img_b]:
                        for idx, i in enumerate(np.arange(-1, 1, bin_size)):
                            gt = tf.greater(img_chan, i)
                            leq = tf.less_equal(img_chan, i + bin_size)
                            # Put together with logical_and, cast to float and sum up entries -> gives count for current bin.
                            hist_entries.append(tf.reduce_sum(tf.cast(tf.logical_and(gt, leq), tf.float32)))

                    # Pack scalars together to a tensor, then normalize histogram.
                    hist = tf.nn.l2_normalize(tf.pack(hist_entries), 0)
                    histograms.append(hist)

            return tf.concat(resized_images, 2), label

        # ----------------------------------------------------------------------
        # DATASET CREATION using tf.contrib.data.Dataset
        # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/data

        # The tf.contrib.data.Dataset framework uses queues in the background to feed in
        # data to the model.
        # We initialize the dataset with a list of filenames and labels, and then apply
        # the preprocessing functions described above.
        # Behind the scenes, queues will load the filenames, preprocess them with multiple
        # threads and apply the preprocessing in parallel, and then batch the data

        # Training dataset
        train_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), train_pairs))
        train_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), train_pairs))
        train_pairs_1 = tf.constant(train_pairs_1)
        train_pairs_2 = tf.constant(train_pairs_2)
        train_labels = tf.constant(train_labels)
        train_dataset = tf.contrib.data.Dataset.from_tensor_slices((train_pairs_1, train_pairs_2, train_labels))
        train_dataset = train_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        train_dataset = train_dataset.shuffle(buffer_size=10000)  # don't forget to shuffle
        batched_train_dataset = train_dataset.batch(args.batch_size)

        # Validation dataset
        val_pairs_1 = list(map(lambda pair: "images/" + pair[0].imageFilename(), val_pairs))
        val_pairs_2 = list(map(lambda pair: "images/" + pair[1].imageFilename(), val_pairs))
        val_pairs_1 = tf.constant(val_pairs_1)
        val_pairs_2 = tf.constant(val_pairs_2)
        val_labels = tf.constant(val_labels)
        val_dataset = tf.contrib.data.Dataset.from_tensor_slices((val_pairs_1, val_pairs_2, val_labels))
        val_dataset = val_dataset.map(_parse_function,
            num_threads=args.num_workers, output_buffer_size=args.batch_size)
        batched_val_dataset = val_dataset.batch(args.batch_size)


        # Now we define an iterator that can operator on either dataset.
        # The iterator can be reinitialized by calling:
        #     - sess.run(train_init_op) for 1 epoch on the training set
        #     - sess.run(val_init_op)   for 1 epoch on the valiation set
        # Once this is done, we don't need to feed any value for images and labels
        # as they are automatically pulled out from the iterator queues.

        # A reinitializable iterator is defined by its structure. We could use the
        # `output_types` and `output_shapes` properties of either `train_dataset`
        # or `validation_dataset` here, because they are compatible.
        iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types,
                                                           batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        # Indicates whether we are in training or in test mode
        is_training = tf.placeholder(tf.bool)

        # Added: an additional layer taking our input tensors and reshaping them
        W_pre = tf.get_variable("W_pre", shape=[7,7,6,3])
        b_pre = tf.get_variable("b_pre", shape=[3])
        pre_out = tf.nn.conv2d(images, W_pre, strides=[1,1,1,1], padding='SAME') + b_pre
        pre_out = tf.nn.relu(pre_out)


        # ---------------------------------------------------------------------
        # Now that we have set up the data, it's time to set up the model.
        # For this example, we'll use VGG-16 pretrained on ImageNet. We will remove the
        # last fully connected layer (fc8) and replace it with our own, with an
        # output size num_classes=8
        # We will train the entire model on our dataset for a few epochs.

        # Get the pretrained model, specifying the num_classes argument to create a new
        # fully connected replacing the last one, called "vgg_16/fc8"
        # Each model has a different architecture, so "vgg_16/fc8" will change in another model.
        # Here, logits gives us directly the predicted scores we wanted from the images.
        # We pass a scope to initialize "vgg_16/fc8" weights with he_initializer
        vgg = tf.contrib.slim.nets.vgg
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=args.weight_decay)):
            logits, _ = vgg.vgg_16(pre_out, num_classes=num_classes, is_training=is_training,
                                   dropout_keep_prob=args.dropout_keep_prob)

        # ---------------------------------------------------------------------
        # Using tf.losses, any loss is added to the tf.GraphKeys.LOSSES collection
        # We can then call the total loss easily
        tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        loss = tf.losses.get_total_loss()

        # Then we want to finetune the entire model for a few epochs.
        # We run minimize the loss only with respect to all the variables.
        full_optimizer = tf.train.GradientDescentOptimizer(args.learning_rate)
        full_train_op = full_optimizer.minimize(loss)

        # Evaluation metrics
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # --------------------------------------------------------------------------
    # Now that we have built the graph and finalized it, we define the session.
    # The session is the interface to *run* the computational graph.
    # We can call our training operations with `sess.run(train_op)` for instance
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        # Train the entire model for a few more epochs, continuing with the *same* weights.
        for epoch in range(args.num_epochs):
            print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs))
            sess.run(train_init_op)
            while True:
                try:
                    _ = sess.run(full_train_op, {is_training: True})
                except tf.errors.OutOfRangeError:
                    break

            # Check accuracy on the train and val sets every epoch
            train_acc = check_accuracy(sess, correct_prediction, is_training, train_init_op)
            val_acc = check_accuracy(sess, correct_prediction, is_training, val_init_op)
            print('Train accuracy: %f' % train_acc)
            print('Val accuracy: %f\n' % val_acc)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
