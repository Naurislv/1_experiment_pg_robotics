"""Sequence Feature Generation model.

MaxoutCNN Neural Network.
"""

import logging
import tensorflow as tf
import tensorflow.contrib.slim as slim  # pylint: disable=E0611


def guntis_net(input_shape, dtype, classes):
    """Maxout CNN Network.

    Input :
        extract_features: Boolean. If True then return features from 4th
        convolution layer with size output_size
        *output_size: must be greater than batch_size. Will padd with 0 to
        get this size.

    """

    ret = {}

    # Input data.

    inputs = tf.placeholder(shape=[None] + input_shape, name='Input', dtype=dtype)
    ret['inputs'] = inputs

    logging.debug('inputs %s', inputs.get_shape().as_list())

    net = slim.conv2d(
        inputs=inputs,
        num_outputs=16,
        kernel_size=[8, 8],
        stride=[4, 4],
        padding='VALID',
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01)
    )

    net = slim.conv2d(
        inputs=net,
        num_outputs=32,
        kernel_size=[4, 4],
        stride=[2, 2],
        padding='VALID',
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01)
    )

    net = slim.fully_connected(
        slim.flatten(net),
        256,
        activation_fn=tf.nn.relu,
        weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01)
    )

    # Output layers for policy and value estimations
    logits = slim.fully_connected(
        net,
        classes,
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(mean=0, stddev=0.01)
    )

    ret['logits'] = logits
    logging.debug('logits %s', logits.get_shape().as_list())

    ret['logits_softmax'] = tf.nn.softmax(logits)

    return ret

def _tf_selu(input_tensor):
    """Tensorflow implementation of SELU.

    Self Normalizing Exponential linear unit

    https://arxiv.org/abs/1706.02515

    https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.050700987355480493419334985294

    res = tf.nn.elu(input_tensor)

    return scale * tf.where(input_tensor > 0, res, alpha * res)
