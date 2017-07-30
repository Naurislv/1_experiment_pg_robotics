"""Sequence Feature Generation model.

MaxoutCNN Neural Network.
"""

import logging
import tensorflow as tf


def build(input_shape, dtype, classes):
    """Maxout CNN Network.

    Input :
        extract_features: Boolean. If True then return features from 4th
        convolution layer with size output_size
        *output_size: must be greater than batch_size. Will padd with 0 to
        get this size.

    """

    ret = {}

    # Input data.
    num_channels = input_shape[-1]

    inputs = tf.placeholder(shape=[None] + input_shape, name='Input', dtype=dtype)
    ret['inputs'] = inputs

    # Variables
    # Convolution part

    w_1 = tf.Variable(
        tf.truncated_normal([15, 15, num_channels, 20], stddev=0.001, dtype=dtype),
        name='w1'
    )
    b_1 = tf.Variable(tf.zeros([20], dtype=dtype), name='b1')

    w_2 = tf.Variable(
        tf.truncated_normal([15, 15, 10, 30], stddev=0.001, dtype=dtype),
        name='w2'
    )
    b_2 = tf.Variable(tf.constant(1.0, shape=[30], dtype=dtype), name='b2')

    # For maxout output of conv layer Depth must be Features % classes == 0
    w_3 = tf.Variable(
        tf.truncated_normal([19, 19, 15, classes * 2], stddev=0.001, dtype=dtype),
        name='w3'
    )
    b_3 = tf.Variable(tf.constant(1.0, shape=[classes * 2], dtype=dtype), name='b3')

    # w_4 = tf.Variable(tf.truncated_normal([16, 16, 128, 512], stddev=0.1), name='w4')
    # b_4 = tf.Variable(tf.constant(1.0, shape=[512]), name='b4')

    # w_5 = tf.Variable(tf.truncated_normal([13, 13, 256, 512], stddev=0.1), name='w5')
    # b_5 = tf.Variable(tf.constant(1.0, shape=[512]), name='b5')

    # w_6 = tf.Variable(tf.truncated_normal([1, 1, 256, 144], stddev=0.1), name='w6')
    # b_6 = tf.Variable(tf.constant(1.0, shape=[144]), name='b6')

    def maxout(inputs, num_units, axis=None):
        """TF Maxout layer implementation."""

        shape = inputs.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1
        if axis is None:  # Assume that channel is the last dimension
            axis = -1
        num_channels = shape[axis]
        if num_channels % num_units:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_units({})'.format(num_channels, num_units))
        shape[axis] = num_units
        shape += [num_channels // num_units]
        outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)

        return outputs

    # Model.
    def model(data):
        """TF Model definition."""

        c_1 = tf.nn.conv2d(data, w_1, [1, 2, 2, 1], padding='VALID')
        logging.debug('c1 %s', c_1.get_shape().as_list())
        h_1 = maxout(c_1 + b_1, 10)  # maxout layer 1
        logging.debug('h1 %s', h_1.get_shape().as_list())

        c_2 = tf.nn.conv2d(h_1, w_2, [1, 1, 1, 1], padding='VALID')
        logging.debug('c2 %s', c_2.get_shape().as_list())
        h_2 = maxout(c_2 + b_2, 15)  # maxout layer 2
        logging.debug('h2 %s', h_2.get_shape().as_list())

        c_3 = tf.nn.conv2d(h_2, w_3, [1, 1, 1, 1], padding='VALID')
        logging.debug('c3 %s', c_3.get_shape().as_list())
        h_3 = maxout(c_3 + b_3, classes)  # maxout layer 3
        logging.debug('h3 %s', h_3.get_shape().as_list())

        return tf.contrib.layers.flatten(h_3)

        # c_4 = tf.nn.conv2d(h_3, w_4, [1, 1, 1, 1], padding='VALID')
        # logging.debug('c4 %s', c_4.get_shape().as_list())
        # h_4 = maxout(c_4 + b_4, 256)  # maxout layer 4
        # logging.debug('h4 %s', h_4.get_shape().as_list())

        # c_5 = tf.nn.conv2d(h_4, w_5, [1, 1, 1, 1], padding='VALID')
        # logging.debug('c5 %s', c_5.get_shape().as_list())
        # h_5 = maxout(c_5 + b_5, 256)  # maxout layer 4
        # logging.debug('h5 %s', h_5.get_shape().as_list())

        # c_6 = tf.nn.conv2d(h_5, w_6, [1, 1, 1, 1], padding='VALID')
        # logging.debug('c6 %s', c_6.get_shape().as_list())
        # h_6 = maxout(c_6 + b_6, 2)  # maxout layer 5
        # logging.debug('h6 %s', h_6.get_shape().as_list())

        # return tf.contrib.layers.flatten(h_6)

    # Training computation.
    logits = model(inputs)
    logging.debug('logits %s', logits.get_shape().as_list())

    ret['logits'] = logits

    logits_softmax = tf.nn.softmax(logits)
    ret['logits_softmax'] = logits_softmax

    return ret
