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
from PolicyGradient import Policy
# from Utils import save_im

# Other imports
import gym
from gym import wrappers
import imageio
import tensorflow as tf
import numpy as np
# import cv2

# Disable TF ERROR messages (Also all other messages). Reason: CUDA_ERROR_OUT_OF_MEMORY
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '6'

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
_FLAGS.DEFINE_string('name', '',
                     'Name of run, will be used to save and load checkpoint and statistic files.')

class GymEnv(object):
    """OpenAI Gym Virtual Environment - setup."""

    def __init__(self, env_name, policy):

        self.data_type = {'tf': tf.float32, 'np': np.float32}

        # To store data and make plots later
        self.episdod_reward = []  # episode reward sum
        self.no_frames = []  # number of frames in all episode

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
            self.policy = policy(list(os_sample_shape),
                                 self.data_type,
                                 self.n_actions)
            self.policy.build('PolicyGradient')

    def run(self, run_name):
        """Start loop."""

        if run_name == '':
            run_name = str(int(round(time.time() * 1000)))

        if not os.path.exists('outputs/' + run_name):
            LOGGER.info('Creating outputs/' + run_name)
            os.makedirs('outputs/' + run_name)

        chk_file = 'outputs/' + run_name + '/model'
        statistics_file = 'outputs/' + run_name + '/statistics.npy'
        statistics = np.array([np.array([]), np.array([])])

        # Try to load tensorflow model
        try:
            self.policy.load(chk_file)
        except tf.errors.NotFoundError:
            LOGGER.warning("Checkpoint Not Found: %s", chk_file)
            self.policy.init_weights()

        # Try to load statistics
        try:
            statistics = np.load(statistics_file)
            LOGGER.info("Statistics loaded %s", statistics.shape)
        except FileNotFoundError:
            LOGGER.warning("Statistics Not Found: %s", statistics_file)

        try:
            self._run()
        except KeyboardInterrupt:  # Stop learning by CTRL + C and save filess
            if not FLAGS.render:
                self.policy.save(chk_file)

                epr = np.concatenate((statistics[0], np.array(self.episdod_reward)), axis=0)
                nfr = np.concatenate((statistics[1], np.array(self.no_frames)), axis=0)

                statistics = np.array([epr, nfr])

                np.save(statistics_file, statistics)
                LOGGER.info("Statistics saved %s %s", statistics_file, statistics.shape)


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

        while True:
            if FLAGS.render:
                self.env.render()

            # preprocess the observation, set input to network to be difference image
            policy_input, prev_img = self.prepro(observation, prev_img)

            # save_im('save_im/{}.jpg'.format(n_frames),
            #         np.concatenate((prev_img, policy_input[:, :, 0]), axis=1))

            action = self.policy.predict(policy_input)[0][0]
            # aprob = aprob[0, :]

            # action = np.random.choice(self.n_actions, p=aprob)
            # label = np.zeros_like(aprob)
            # label[action] = 1

            label = np.zeros((self.n_actions), dtype=self.data_type['np'])
            label[action] = 1

            # step the environment and get new measurements
            observation, reward, done, _ = self.env.step(action)
            reward_sum += reward

            if n_frames > 2:
                # Start 'recording' only after 3 frame because (e.g. for Pong) observation
                # from env.reset() differs from the one which comes from env.setp()
                obs_his.append(policy_input)
                action_his.append(label)

                # record reward (has to be done after we call step() to get reward
                # for previous action)
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

                if episode_number % batch_size == 0:
                    action_counter = [np.where(aprob == 1)[0][0] for aprob in action_his]
                    action_space = {i: 0 for i in range(self.n_actions)}
                    for act in action_counter:
                        action_space[act] += 1

                    LOGGER.info("Update weights from %d frames with average score: %s",
                                len(reward_his), sum(reward_his) / batch_size)
                    LOGGER.info("Used action space: %s", action_space)

                    # compute the discounted reward backwards through time
                    reward_his = self.discount_rewards(np.array(reward_his))
                    # standardize the rewards to be unit normal
                    # (helps control the gradient estimator variance)
                    reward_his -= np.mean(reward_his)
                    tmp = np.std(reward_his)
                    if tmp > 0:
                        reward_his /= tmp  # fix zero-divide

                    self.policy.fit(np.array(obs_his),
                                    np.vstack(action_his),
                                    np.vstack(reward_his))

                    # Reset history
                    reward_his = []  # save rewards
                    action_his = []  # action history
                    obs_his = []  # save eposiodes

                # Just for plotting
                self.episdod_reward.append(reward_sum)
                self.no_frames.append(n_frames)

                observation = self.env.reset()  # reset env
                reward_sum = 0
                n_frames = 1
                prev_img = None
                episode_number += 1
                start_time = time.time()

    def prepro(self, img, prev_img):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        # Resize image using interpolation
        # img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        # downsample by factor of 2 ##choose colour 2 to improve visibility in most games
        img = img[::2, ::2, 2]

        # img[img == 17] = 0 # erase background (background type 1)
        # img[img == 192] = 0 # erase background (background type 2)
        # img[img == 136] = 0 # erase background (background type 3)
        # img[img != 0] = 1 # everything else (paddles, ball) just set to 1

        img = img[17:96, :]

        img = img.astype(self.data_type['np'])

        # Insert motion in frame by subtracting previous frame from current
        if prev_img is not None:
            # absdiff does not give different colors if differences
            # so neural network can't distinguish betwwen previous and current frame
            # policy_input = cv2.absdiff(img, prev_img)
            policy_input = img - prev_img
        else:
            policy_input = np.zeros_like(img)

        prev_img = img

        policy_input = np.expand_dims(policy_input, -1)

        return policy_input, prev_img

    def discount_rewards(self, reward_his, gamma=.99):
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

    ENV = GymEnv('Pong-v4', Policy)

    ENV.run(run_name=FLAGS.name)
