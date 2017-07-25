"""Sample Tensorflow network for Policy Gradient.

Policy Gradient in TF:
    https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
"""

# Standard imports
import logging
import time

# Other imports
import tensorflow as tf
import tensorflow.contrib.slim as slim  # pylint: disable=E0611


TF_CONFIG = tf.ConfigProto()
TF_CONFIG.allow_soft_placement = True
TF_CONFIG.gpu_options.allocator_type = 'BFC'  # pylint: disable=E1101
# TF_CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.8
TF_CONFIG.gpu_options.allow_growth = True  # pylint: disable=E1101
# TF_CONFIG.log_device_placement = True

class SampleNet(object):
    """Three Dense layers"""

    def __init__(self, state_shape):
        """
        ARGS:
            state_shape: list of input shape for convolutional network e.g. [65, 65, 3]
            scope: variable scope, with this scope you will be able to access those variables

        """
        self.rmsprop_decay = 0.99
        self.learning_rate = 0.1

        self.state_shape = state_shape

        self._sess = tf.Session(config=TF_CONFIG)

        self.inputs = None
        self.logits = None

        self.sample = None
        self.actions = None
        self.advantages = None

        self.loss = None
        self.train = None

    def build(self, scope):
        """Build graph."""

        with tf.variable_scope(scope):  # pylint: disable=E1129
            logging.info('Building Network in %s scope', scope)

            self.inputs = tf.placeholder(shape=[None] + self.state_shape, dtype=tf.float32)

            net = slim.conv2d(
                activation_fn=self._tf_selu,
                inputs=self.inputs,
                num_outputs=24,
                kernel_size=[5, 5],
                stride=[2, 2],
                padding='VALID',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
            )

            # net = slim.conv2d(activation_fn=self._tf_selu,
            #                   inputs=net, num_outputs=36,
            #                   kernel_size=[5, 5], stride=[2, 2], padding='VALID',
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

            # net = slim.conv2d(activation_fn=self._tf_selu,
            #                   inputs=net, num_outputs=48,
            #                   kernel_size=[5, 5], stride=[2, 2], padding='VALID',
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

            net = slim.conv2d(
                activation_fn=self._tf_selu,
                inputs=net,
                num_outputs=36,
                kernel_size=[3, 3], stride=[1, 1],
                padding='VALID',
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
            )

            # net = slim.conv2d(activation_fn=self._tf_selu,
            #                   inputs=net, num_outputs=48,
            #                   kernel_size=[2, 2], stride=[1, 1], padding='VALID',
            #                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))

            net = slim.fully_connected(
                slim.flatten(net),
                50,
                activation_fn=self._tf_selu,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
            )

            net = slim.fully_connected(
                net,
                10,
                activation_fn=self._tf_selu,
                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
            )

            # net = slim.fully_connected(
            #     net,
            #     10,
            #     activation_fn=self._tf_selu,
            #     weights_initializer=tf.truncated_normal_initializer(stddev=0.01)
            # )

            self.logits = slim.fully_connected(net, 2)

            # Samples an action from multinomial distribution
            self.sample = tf.transpose(tf.multinomial(self.logits, 1))

            log_prob = tf.log(tf.nn.softmax(self.logits))

            # training part of graph
            self.actions = tf.placeholder(tf.int32)
            self.advantages = tf.placeholder(tf.float32)

            # get log probs of actions from episode
            indices = tf.range(0, tf.shape(log_prob)[0]) * tf.shape(log_prob)[1] + self.actions
            act_prob = tf.gather(tf.reshape(log_prob, [-1]), indices)

            # surrogate loss
            self.loss = - tf.reduce_sum(tf.multiply(act_prob, self.advantages))  # pylint: disable=E1130

            # update
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train = optimizer.minimize(self.loss)

    def init_weights(self):
        """Initialize TF variables."""

        logging.info('Initializing Network variables')
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)

    def fit(self, obs, actions, advantages):
        """Train neural network."""


        batch_feed = {self.inputs: obs,
                      self.actions: actions,
                      self.advantages: advantages}

        _, loss = self._sess.run([self.train, self.loss], feed_dict=batch_feed)
        logging.info("Loss : %f", loss)

    def predict(self, observation):
        """Predict action from observation.

        Return index of action.
        """
        return self._sess.run(self.sample, feed_dict={self.inputs: observation})

    def save(self, path):
        """Save Tesnroflow checkpoint."""
        saver = tf.train.Saver()

        path = path + '_' + str(int(round(time.time() * 1000)))
        saver.save(self._sess, path)
        logging.info('Checkpoint saved %s', path)

    def load(self, path):
        """Save Tesnroflow checkpoint."""
        saver = tf.train.Saver()

        saver.restore(self._sess, path)
        logging.info('Variables loaded from checkpoint file: %s', path)

    def _tf_selu(self, input_tensor):
        """Tensorflow implementation of SELU.

        Self Normalizing Exponential linear unit

        https://arxiv.org/abs/1706.02515

        https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py
        """
        alpha = 1.6732632423543772848170429916717
        scale = 1.050700987355480493419334985294

        res = tf.nn.elu(input_tensor)

        return scale * tf.where(input_tensor > 0, res, alpha * res)
