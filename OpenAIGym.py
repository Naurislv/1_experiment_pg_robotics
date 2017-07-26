"""Train formula to drive a track.

Deep Reinforcement Learning using OpenAI Gym CarRacing-v0 as environment.

Run for more info :

    python CarRacing.py --help

SUPPORTED ENVIRONMENTS:

    1.CarRacing-v0 : https://gym.openai.com/envs/CarRacing-v0

        Environment input parameters is numpy array as follows:

            [arg1 arg2 arg3]

            arg1 : angle [-1..1]
            arg2 : speed [0..1]
            arg3 : brakes [0..1]

        Environment outputs :

            observation : (96, 96, 3) image
                        Some indicators shown at the bottom of the window and the state RGB
                        buffer. From left to right: true speed, four ABS sensors, steering
                        wheel position, gyroscope.

            reward : amount of reward achieved by previous action. Goal is to increase reward.
                    -0.1 for every frame and +1000/N for every track tile visited, where N
                    is total number of tiles in track. For example, if you have finished in
                    732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

            done : boolean. Indicates weather to reset environment.

Insipred from :
    http://karpathy.github.io/2016/05/31/rl/
    https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

"""

# Standard imports
import time
import logging
import os

# Local imports
# For each worker there is seperate AC3Network therefore scopes / names of
# networks has to be handled correctly
from Policy import SampleNet
# from Utils import save_im

# Other imports
import gym
from gym import wrappers
import imageio
import tensorflow as tf
import numpy as np

# Disable TF ERROR messages (Also all other messages). Reason: CUDA_ERROR_OUT_OF_MEMORY
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get gym logger
LOGGER = logging.getLogger()
LOGGER.setLevel(logging.DEBUG)

# Create tensorflow flags for input arguments
_FLAGS = tf.app.flags
FLAGS = _FLAGS.FLAGS

_FLAGS.DEFINE_boolean('video_small', False, "Whether to record and save video or not. Boolean.")
_FLAGS.DEFINE_boolean('video_big', False, "Whether to record and save video or not. Gym built "
                                          "in command. Also create report .json files. Boolean.")
_FLAGS.DEFINE_boolean('render', True, "Whether to render video for visual or not.")
_FLAGS.DEFINE_boolean('gpu', False, "Use GPU.")

class GymEnv(object):
    """OpenAI Gym Virtual Environment - setup."""

    def __init__(self, env_name, policy_net):

        self.data_type = {'tf': tf.float32, 'np': np.float32}

        # Create gym environment for CarRacing-v0
        self.env = gym.make(env_name)
        LOGGER.info("%s initialized", env_name)

        if FLAGS.video_big:
            self.env = wrappers.Monitor(self.env,
                                        '{}-{}'.format(env_name, int(round(time.time() * 1000))))

        if FLAGS.video_small:
            self.writer = imageio.get_writer('{}.mp4'.format(env_name), mode='I')

        self.n_actions = self.env.action_space.n
        LOGGER.info('Action Space %s total of %d actions',
                    self.env.action_space, self.n_actions)
        LOGGER.info([func for func in dir(self.env.action_space) if '__' not in func])
        LOGGER.info('Observation Space %s', self.env.observation_space)
        LOGGER.info([func for func in dir(self.env.observation_space) if '__' not in func])

        for _ in range(10):
            LOGGER.info('AS sample: %s', self.env.action_space.sample())

        # Observation sapce shape after preprocessing
        os_sample_shape = self.prepro(self.env.observation_space.sample(), None)[0].shape
        LOGGER.info("Observation sample shape after preprocess: %s", os_sample_shape)
        # Policy network - responsilbe for sampled actions
        if FLAGS.gpu:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            self.policy_net = policy_net(list(os_sample_shape),
                                         self.data_type,
                                         self.n_actions)
            self.policy_net.build('PolicyGradient')

    def run(self, chk_path_load='', chk_path_save='model'):
        """Start loop."""

        try:
            self.policy_net.load(chk_path_load)
        except Exception as e:
            self.policy_net.init_weights()

        try:
            self._run()
        except KeyboardInterrupt:
            self.policy_net.save(chk_path_save)

    def _run(self):
        """Run virtual environment loop."""

        # 1 episode is multiple games (until game env responde with done == True)
        episode_number = 1
        n_frames = 1  # frames per episode
        prev_img = None  # Previous image

        reward_sum = 0
        batch_size = 10

        # Used for training after episode
        reward_his = []  # save rewards
        action_his = []  # action history
        obs_his = []  # save eposiodes

        observation = self.env.reset()

        start_time = time.time()
        train_time = start_time

        lives = None

        while True:
            if FLAGS.render:
                self.env.render()

            # preprocess the observation, set input to network to be difference image
            policy_input, prev_img = self.prepro(observation, prev_img)

            # save_im('save_im/{}.jpg'.format(n_frames),
            #         np.concatenate((prev_img, policy_input), axis=1))

            policy_input_expand = np.expand_dims(policy_input, axis=0)
            action_idx = self.policy_net.predict(policy_input_expand)[0, :]

            label = np.zeros((self.n_actions), dtype=self.data_type['np'])
            label[action_idx] = 1

            # step the environment and get new measurements
            observation, reward, done, info = self.env.step(action_idx)
            reward_sum += reward

            if lives is None:
                lives = info['ale.lives']
            if info['ale.lives'] < lives:
                done = True # End game when loose first live
                LOGGER.info("End game because lost first live.")

            # record reward (has to be done after we call step() to get reward for previous action)
            obs_his.append(policy_input)
            action_his.append(label)
            reward_his.append(reward)

            if n_frames % 50 == 0:
                end_time = time.time()

                fps = 50 / (end_time - start_time)
                LOGGER.debug("%s. [%.2fs] FPS: %.2f, Reward Sum: %s",
                             episode_number, end_time - train_time, fps, reward_sum)
                start_time = time.time()

            if FLAGS.video_small:
                self.writer.append_data(observation)

            n_frames += 1

            if done:  # When Game env say it's done - end of episode.
                LOGGER.info('')
                LOGGER.info("Episode done! Reward sum: %.2f , Frames: %d",
                            reward_sum, n_frames)
                LOGGER.info('')

                n_frames = 1

                if episode_number % batch_size == 0:
                    LOGGER.info("Update weights from %d frames with average score: %s",
                                len(reward_his), sum(reward_his) / batch_size)

                    # compute the discounted reward backwards through time
                    discounted_reward = self.discount_rewards(np.array(reward_his))
                    # standardize the rewards to be unit normal
                    # (helps control the gradient estimator variance)
                    discounted_reward -= np.mean(discounted_reward)
                    tmp = np.std(discounted_reward)
                    if tmp > 0:
                        discounted_reward /= tmp  # fix zero-divide

                    self.policy_net.fit(np.array(obs_his),
                                        np.vstack(action_his),
                                        np.vstack(discounted_reward))

                    # Reset history
                    reward_his = []  # save rewards
                    action_his = []  # action history
                    obs_his = []  # save eposiodes

                observation = self.env.reset()  # reset env
                reward_sum = 0
                lives = None
                prev_img = None
                episode_number += 1
                start_time = time.time()

    def prepro(self, img, prev_img):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        img = img[::2, ::2, 0] # downsample by factor of 2
        img = img[17:97, :]

        img = img.astype(self.data_type['np'])

        if prev_img is not None:
            policy_input = img - prev_img
        else:
            policy_input = np.zeros_like(img)

        prev_img = img

        return policy_input, prev_img

    def discount_rewards(self, reward_his, gamma=0.99):
        """Returns discounted rewards
        Args:
            reward_his (1-D array): a list of `reward` at each time step
            gamma (float): Will discount the future value by this rate
        Returns:
            discounted_r (1-D array): same shape as input `R`
                but the values are discounted
        Examples:
            >>> R = [1, 1, 1]
            >>> discount_rewards(R, .99) # before normalization
            [1 + 0.99 + 0.99**2, 1 + 0.99, 1]
        """

        discounted_r = np.zeros_like(reward_his)
        running_add = 0
        for i in reversed(range(0, reward_his.size)):
            # reset the sum, since this was a game boundary (pong specific!)
            if reward_his[i] != 0:
                running_add = 0
            running_add = running_add * gamma + reward_his[i]
            discounted_r[i] = running_add

        return discounted_r

if __name__ == "__main__":

    ENV = GymEnv('Pong-v4', SampleNet)

    ENV.run(chk_path_load='/home/nauris/Dropbox/coding/openai_gym/models/model_1501094533767',
            chk_path_save='/home/nauris/Dropbox/coding/openai_gym/models/model')
