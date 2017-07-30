"""Sample Tensorflow network for Policy Gradient.

Policy Gradient in TF:
    https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
"""

# Standard imports
import logging

# Other imports
import tensorflow as tf
import numpy as np
# from MaxoutNet import maxout_cnn as policy_net
# from GuntisNet import guntis_net as policy_net
from KarpathyNet import karpathy_net as policy_net

# TF_CONFIG = tf.ConfigProto()
# TF_CONFIG.allow_soft_placement = True
# TF_CONFIG.gpu_options.allocator_type = 'BFC'  # pylint: disable=E1101
# TF_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.8
# TF_CONFIG.gpu_options.allow_growth = True  # pylint: disable=E1101
# TF_CONFIG.log_device_placement = True

class Policy(object):
    """Three Dense layers"""

    def __init__(self, state_shape, data_type, n_actions):
        """
        ARGS:
            state_shape: list of input shape for convolutional network e.g. [65, 65, 3]
            scope: variable scope, with this scope you will be able to access those variables

        """
        self.rmsprop_decay = 0.99
        self.learning_rate = 1e-3
        self.state_shape = state_shape

        self._sess = tf.Session()

        self.net = None
        self.actions = None
        self.advantages = None
        self.sample = None
        self.loss = None
        self.train = None

        self.data_type = data_type
        self.n_actions = n_actions

    def build(self, scope):
        """Build graph."""

        with tf.variable_scope(scope):  # pylint: disable=E1129
            logging.info('Building Network in %s scope', scope)

            self.net = policy_net(self.state_shape, self.data_type['tf'], self.n_actions)

            self.actions = tf.placeholder(dtype=self.data_type['tf'], shape=[None, self.n_actions])
            self.advantages = tf.placeholder(dtype=self.data_type['tf'], shape=[None, 1])

            # Samples an action from multinomial distribution
            self.sample = tf.transpose(tf.multinomial(self.net['logits'], 1))

            # surrogate loss
            self.loss = - tf.reduce_sum(self.advantages *  # pylint: disable=E1130
                                        self.actions *
                                        tf.log(self.net['logits_softmax'] + 0.0001))

            # update
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.learning_rate,
                decay=self.rmsprop_decay
            )

            # Discussions - how to reduce memory consumpsion in TF:
            # https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/q9bT3Ql2bYk
            # https://stackoverflow.com/questions/36194394/how-i-reduce-memory-consumption-in-a-loop-in-tensorflow
            self.train = optimizer.minimize(self.loss)

    def init_weights(self):
        """Initialize TF variables."""

        logging.info('Initializing network Random variables')
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

    def fit(self, obs, actions, advantages):
        """Train neural network."""

        batch_feed = {self.net['inputs']: obs,
                      self.actions: actions,
                      self.advantages: advantages}

        _, loss = self._sess.run([self.train, self.loss], feed_dict=batch_feed)
        logging.info("Loss sum: %s", loss)

    def predict(self, observation):
        """Predict action from observation.

        Return index of action.
        """

        observation = np.expand_dims(observation, axis=0)
        return self._sess.run(self.sample, feed_dict={self.net['inputs']: observation})

    def save(self, path):
        """Save Tesnroflow checkpoint."""
        saver = tf.train.Saver()

        saver.save(self._sess, path)
        logging.info('Checkpoint saved %s', path)

    def load(self, path):
        """Save Tesnroflow checkpoint."""
        saver = tf.train.Saver()

        saver.restore(self._sess, path)
        logging.info('Variables loaded from checkpoint file: %s', path)
