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
from SimpleClassificator import SimpleNet
from Utils import save_im

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

class GymEnv(object):
    """OpenAI Gym Virtual Environment - setup."""

    def __init__(self, env_name, policy_net):

        self.gamma = 0.99 # discount factor for reward

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
        # Policy network - responsilbe for sampled actions
        os_sample_shape = (os_sample_shape[1], os_sample_shape[2], os_sample_shape[3])
        self.policy_net = policy_net(os_sample_shape)

    def run(self):
        """Run virtual environment loop."""

        # 1 episode is multiple games (until game env responde with done == True)
        episode_number = 0
        n_frames = 0  # frames per episode
        prev_img = None  # Previous image

        reward_sum = 0

        # Used for training after episode
        reward_his = []  # save rewards
        labels_his = []  # save labels
        obs_his = []  # save eposiodes

        observation = self.env.reset()

        while True:
            if FLAGS.render:
                self.env.render()

            # preprocess the observation, set input to network to be difference image
            policy_input, prev_img = self.prepro(observation, prev_img)

            # save_im('save_im/{}.jpg'.format(counter), policy_input)

            policy_output = self.policy_net.predict(policy_input)

            action_prob = policy_output[0][0]

            # Generate proper action for game env based on neural network output probability
            # and give a chance to do random move
            action = 2 if np.random.uniform() < action_prob else 3  # roll the dice!

            obs_his.append(policy_input)

            # Basically generate label for neural network. Binary label 0 or 1 for classification.
            label = 1 if action == 2 else 0  # a "fake label"
            labels_his.append(label)

            # step the environment and get new measurements
            observation, reward, done, _ = self.env.step(action)
            reward_sum += reward

            # record reward (has to be done after we call step() to get reward for previous action)
            reward_his.append(reward)

            if n_frames % 40 == 0:
                LOGGER.debug("Every 40th frame. Episode: %s. Obs shape: %s Reward Sum: %s,"
                             "Policy: %s, Action: %s, Done: %s", episode_number,
                             observation.shape, reward_sum, policy_output, action, done)

            if FLAGS.video_small:
                self.writer.append_data(observation)

            n_frames += 1

            if done:  # When Game env say it's done - it's done!
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

    def prepro(self, img, prev_img):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        # img = img[0:83] # crop

        policy_input = img - prev_img if prev_img is not None else np.zeros(img.shape)
        prev_img = img

        policy_input = np.expand_dims(policy_input, axis=0)

        return policy_input, prev_img

    def discount_rewards(self, reward_his):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(reward_his)
        running_add = 0
        for i in reversed(range(0, reward_his.size)):
            # reset the sum, since this was a game boundary (pong specific!)
            if reward_his[i] != 0:
                running_add = 0
            running_add = running_add * self.gamma + reward_his[i]
            discounted_r[i] = running_add

        return discounted_r

if __name__ == "__main__":

    ENV = GymEnv('Pong-v0', SimpleNet)

    ENV.run()
