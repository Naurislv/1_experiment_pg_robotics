"""During traning there is information which we need to keep track of and simple data
management is required."""

# Standard imports
import time

# Dependency imports
import numpy as np
import tensorflow as tf

# Local imports
from utils import get_logger, discount_rewards


LOGGER = get_logger('Data')


class DataManager:
    """Record and hold data required during training."""

    def __init__(self, summary_writer):

        self.summary_writer = summary_writer

        self._labels = []
        self._rewards = []
        self._observations = []

        self.record_timestamp = None

        # Load the step from previous trainings
        self.episode_number = tf.summary.experimental.get_step()
        if not self.episode_number:
            self.episode_number = 1

        self.start_time = time.time()
        self.record_counter = 0
        self._last_record_count = 0

    def record_data(self, observation, reward, action, nb_actions):
        """Record data for historical purposes."""

        if not self._rewards:
            self.record_timestamp = time.time()

        self._observations.append(observation)
        self._rewards.append(reward)

        label = np.zeros((nb_actions), dtype=np.float32)
        label[action] = 1

        self._labels.append(label)

        self.record_counter += 1

    @property
    def rewards_discounted(self):
        """Compute the discounted reward backwards through time."""

        reward_his = discount_rewards(self.rewards)
        # standardize the rewards to be unit normal
        # (helps control the gradient estimator variance)
        reward_his -= np.mean(reward_his)
        tmp = np.std(reward_his)
        if tmp > 0:
            reward_his /= tmp  # fix zero-divide

        return reward_his

    @property
    def rewards(self):
        """Return reward history in numpy array."""
        return np.array(self._rewards, dtype=np.float32)

    @property
    def rewards_episode(self):
        """Return reward history in numpy array."""
        return np.array(
            self._rewards[self._last_record_count: self.record_counter], dtype=np.float32)

    @property
    def labels(self):
        """Return reward history in numpy array."""
        return np.array(self._labels, dtype=np.float32)

    @property
    def observations(self):
        """Return reward history in numpy array."""
        return np.array(self._observations, dtype=np.uint8)

    @property
    def record_counter_episode(self):
        """Return record counter for single episode."""
        return self.record_counter - self._last_record_count

    def log_summary(self):
        """Print out in logs summary about training."""

        current_time = time.time()

        fps = 0
        if self.record_counter > 2:  # if observation is not empty
            fps = self.record_counter / (current_time - self.record_timestamp)

        LOGGER.debug("%s. T[%.2fs] FPS: %.2f, Reward Sum: %s",
                     self.episode_number, current_time - self.start_time, fps,
                     sum(self._rewards[self._last_record_count: self.record_counter]))

    def next_batch(self):
        """Clear gathered data and prepare to for next episode."""

        self._labels = []
        self._rewards = []
        self._observations = []
        self.record_counter = 0
        self._last_record_count = 0
        self.record_timestamp = None

    def next_episode(self):
        """Next Episode."""

        episode_rewards = sum(self._rewards[self._last_record_count: self.record_counter])
        episode_record_counter = self.record_counter - self._last_record_count

        with self.summary_writer.as_default():
            tf.summary.scalar('reward', episode_rewards, step=self.episode_number)
            tf.summary.scalar('number of records', episode_record_counter, step=self.episode_number)

        self.episode_number += 1
        self._last_record_count = self.record_counter

        tf.summary.experimental.set_step(self.episode_number)
