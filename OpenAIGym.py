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

# Local imports
# For each worker there is seperate AC3Network therefore scopes / names of
# networks has to be handled correctly
from SamplePolicyGradNet import SampleNet
# from Utils import save_im

# Other imports
import gym
from gym import wrappers
import imageio
import tensorflow as tf
import numpy as np

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

        # Create gym environment for CarRacing-v0
        self.env = gym.make(env_name)
        LOGGER.info("%s initialized", env_name)

        if FLAGS.video_big:
            self.env = wrappers.Monitor(self.env,
                                        '{}-{}'.format(env_name, int(round(time.time() * 1000))))

        if FLAGS.video_small:
            self.writer = imageio.get_writer('{}.mp4'.format(env_name), mode='I')

        LOGGER.info('Action Space %s', self.env.action_space)
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
            self.policy_net = policy_net(list(os_sample_shape))
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
        episode_number = 0
        n_frames = 1  # frames per episode
        prev_img = None  # Previous image

        reward_sum = 0
        batch_size = 10

        # Used for training after episode
        reward_his = []  # save rewards
        action_his = []  # action history
        obs_his = []  # save eposiodes

        reward_his_batch = []
        action_hist_batch = []
        obs_his_batch = []

        observation = self.env.reset()

        start_time = time.time()

        while True:
            if FLAGS.render:
                self.env.render()

            # preprocess the observation, set input to network to be difference image
            policy_input, prev_img = self.prepro(observation, prev_img)
            obs_his.append(policy_input)

            # save_im('save_im/{}.jpg'.format(counter), policy_input)
            policy_input = np.expand_dims(policy_input, axis=0)
            action_idx = self.policy_net.predict(policy_input)[0][0]

            action_dict = {0: 2, 1: 3}
            action = action_dict[action_idx]
            action_his.append(action_idx)

            # step the environment and get new measurements
            observation, reward, done, _ = self.env.step(action)
            reward_sum += reward

            # record reward (has to be done after we call step() to get reward for previous action)
            reward_his.append(reward)

            if n_frames % 50 == 0:
                end_time = time.time()

                fps = 50 / (end_time - start_time)
                LOGGER.debug("%s. FPS: %.2f, Reward Sum: %s",
                             episode_number, fps, reward_sum)
                start_time = time.time()

            if FLAGS.video_small:
                self.writer.append_data(observation)

            n_frames += 1

            if done:  # When Game env say it's done - end of episode.
                episode_number += 1

                logging.info("Done!")
                observation = self.env.reset()  # reset env
                reward_sum = 0
                prev_img = None

                # compute the discounted reward backwards through time
                discounted_reward = self.discount_rewards(np.vstack(reward_his))
                # standardize the rewards to be unit normal
                # (helps control the gradient estimator variance)
                discounted_reward -= np.mean(discounted_reward)
                discounted_reward /= np.std(discounted_reward)

                # advantages = [len(reward_his)] * len(reward_his)

                reward_his_batch.extend(discounted_reward)
                action_hist_batch.extend(action_his)
                obs_his_batch.extend(obs_his)

                # Reset history
                reward_his = []  # save rewards
                action_his = []  # action history
                obs_his = []  # save eposiodes

                n_frames = 1

                if episode_number % batch_size == 0:
                    LOGGER.info("Update weights!")

                    # update policy
                    # normalize rewards; don't divide by 0
                    reward_his_batch = ((reward_his_batch - np.mean(reward_his_batch)) /
                                        (np.std(reward_his_batch) + 1e-10))

                    self.policy_net.fit(obs_his_batch, action_hist_batch, reward_his_batch)

                    reward_his_batch = []
                    action_hist_batch = []
                    obs_his_batch = []

                start_time = time.time()

    def prepro(self, img, prev_img):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """

        img = img[35:195] # crop
        img = img[::2, ::2, 0] # downsample by factor of 2
        img[img == 144] = 0 # erase background (background type 1)
        img[img == 109] = 0 # erase background (background type 2)
        img[img != 0] = 1 # everything else (paddles, ball) just set to 1

        img = img.astype(np.float)
        img = img.reshape((img.shape[0], img.shape[1], 1))

        policy_input = img - prev_img if prev_img is not None else np.zeros(img.shape)
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

    ENV = GymEnv('Pong-v0', SampleNet)

    ENV.run(chk_path_load='/home/nauris/Dropbox/coding/openai_gym/models/model_1500995521623',
            chk_path_save='/home/nauris/Dropbox/coding/openai_gym/models/model')
