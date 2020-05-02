"""Sample Tensorflow network for Policy Gradient.

Policy Gradient in TF:
    https://gist.github.com/shanest/535acf4c62ee2a71da498281c2dfc4f4
"""

# Dependency imports
import tensorflow as tf

# Local imports
from models import NatureCNN, surrogate_loss
from utils import get_logger


LOGGER = get_logger('policy')


class Policy(object):
    """Three Dense layers"""

    def __init__(self, nb_actions, learing_rate=1e-4, decay=0.99):
        """
        ARGS:
            learing_rate: learning rate
            decay: RMSprop decay
        """

        self.nb_actions = nb_actions
        self.model = NatureCNN(self.nb_actions)

        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=learing_rate,
            rho=decay
        )

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, observations, actions, advantages, step=None):
        """Make a single training step."""

        with tf.GradientTape() as tape:

            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(observations, training=True)
            loss = surrogate_loss(predictions, actions, advantages)

        if step is not None:
            tf.summary.scalar('loss', loss, step=step)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return predictions

    @tf.function
    def sample_action(self, observation, test=False):
        """Sample action from observation.

        Return index of action.
        """

        observations = tf.expand_dims(observation, axis=0)
        if not test:
            action = self.model(observations, training=False, sample=True)[0]
        else:
            action = self.model(observations, training=False, sample=False)
            action = tf.argmax(input=action, axis=1)

        return action

    def save(self, checkpoint_path):
        """Save Tesnroflow checkpoint."""

        self.model.save_weights(checkpoint_path)

        LOGGER.info('Checkpoint saved %s', checkpoint_path)

    def load(self, checkpoint_path):
        """Load Tesnsorflow checkpoint."""

        self.model.load_weights(checkpoint_path)

        LOGGER.info('Weights loaded from checkpoint file: %s', checkpoint_path)
