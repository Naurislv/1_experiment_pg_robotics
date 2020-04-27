"""
Some function implementations taken from:
    https://github.com/openai/baselines/blob/b99a73afe37206775ac8b884d32a36e213a3fac2/
    baselines/common/models.py
"""

# Dependency imports
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Softmax
from tensorflow.keras import Model
import numpy as np


def conv(n_f, r_f, stride, activation, pad='valid', init_scale=1.0, data_format='channels_last'):
    """Conv2D wrapper."""

    layer = Conv2D(
        filters=n_f, kernel_size=r_f, strides=stride, padding=pad, activation=activation,
        data_format=data_format, kernel_initializer=ortho_init(init_scale)
    )

    return layer


def ortho_init(scale=1.0):
    """Orthogonal weight initialization"""

    def _ortho_init(shape, dtype, partition_info=None):  # pylint: disable=unused-argument
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a_val = np.random.normal(0.0, 1.0, flat_shape)
        u_val, _, v_val = np.linalg.svd(a_val, full_matrices=False)
        q_val = u_val if u_val.shape == flat_shape else v_val  # pick the one with the correct shape
        q_val = q_val.reshape(shape)
        return (scale * q_val[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


@tf.function(experimental_relax_shapes=True)
def surrogate_loss(logits_softmax, action_hist, reward_hist):
    """Surrogate loss"""

    loss = - tf.reduce_sum(
        tf.math.multiply(
            tf.math.multiply(reward_hist, action_hist), tf.math.log(logits_softmax + 0.0001)))

    return loss


class NatureCNN(Model):
    """Nature CNN Policy network"""

    def __init__(self, nb_outputs):
        super(NatureCNN, self).__init__()

        self.hidden_1 = conv(n_f=32, r_f=8, stride=4, activation='relu', init_scale=np.sqrt(2))
        self.hidden_2 = conv(n_f=64, r_f=4, stride=2, activation='relu', init_scale=np.sqrt(2))
        self.hidden_3 = conv(n_f=64, r_f=3, stride=1, activation='relu', init_scale=np.sqrt(2))

        self.flatten = Flatten()
        self.dense_1 = Dense(
            units=512, kernel_initializer=ortho_init(np.sqrt(2)), activation='relu')

        self.logits = Dense(
            units=nb_outputs, kernel_initializer=ortho_init(np.sqrt(2)))

        self.logits_softmax = Softmax()

    def call(self, inputs, training=True, sample=False):  # pylint: disable=arguments-differ
        """This is a call function which will be executed when model will be called."""

        ret = tf.cast(inputs, tf.float32) / 255.
        ret = self.hidden_1(ret)
        ret = self.hidden_2(ret)
        ret = self.hidden_3(ret)
        ret = self.flatten(ret)
        ret = self.dense_1(ret)
        logits = self.logits(ret)

        if sample:
            return tf.transpose(tf.random.categorical(logits, 1))

        predictions = self.logits_softmax(logits)
        return predictions
