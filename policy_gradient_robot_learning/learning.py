"""Train formula to drive a track.

Deep Reinforcement Learning using OpenAI Gym CarRacing-v0 as environment.

Run for more info :

    python CarRacing.py --help

SAMPLE ENVIRONMENT:

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
import logging
import datetime
from os.path import dirname, join, abspath
import time

# Dependency imports
import gym
import tensorflow as tf
import numpy as np

# Local imports
from policy_gradient import Policy
from argument_parser import user_args
from utils import get_logger, pong_img_preproc
from data_management import DataManager


class TrainingStoppedException(Exception):
    """Used to stop the training and catch to store some files."""


def setup_outputs():
    """Setup Tensorboard and runtime outputs directories and files"""

    if 'gs://' in ARGS.output_dir:
        dir_path = ARGS.output_dir
    else:
        dir_path = join(dirname(abspath(__file__)), ARGS.output_dir)

    prefix = ''
    if ARGS.test:
        prefix = 'test'

    session_descriptor = (f"{prefix}lr[{ARGS.learning_rate}]_bs[{ARGS.batch_size}]_"
                          f"policy[{ARGS.policy}]_model[{ARGS.model}]_env[{ARGS.env}]")

    session_id = ARGS.session_id
    if not session_id:
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = join(dir_path, f'{session_id}/{session_descriptor}/')
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    step_path = join(dir_path, f'{session_id}/{prefix}step.txt')

    try:
        episode_number = int(tf.io.gfile.GFile(step_path).read())
        tf.summary.experimental.set_step(episode_number)
    except tf.errors.NotFoundError:
        pass

    chk_path = join(dir_path, f'{session_id}/model')

    return summary_writer, chk_path


def make_env(name):
    """Create an environment."""

    env = gym.make(name)
    observation = env.reset()
    LOGGER.info("%s initialized", name)

    for _ in range(1):
        observation, _, _, _ = env.step(0)

    return env, observation


def learning(env_name, policy, batch_size, summary_writer):
    """Learning is happening here."""

    data_holder = DataManager(summary_writer)

    env, observation = make_env(env_name)
    prev_observation = None
    prev_reward = None

    while True:
        if ARGS.test:
            env.render()

        # preprocess the observation, set input to network to be difference image
        prev_observation, policy_input = pong_img_preproc(prev_observation, observation)

        action = policy.sample_action(policy_input, test=ARGS.test)[0]

        # step the environment and get new measurements
        observation, reward, done, _ = env.step(action)

        data_holder.record_data(policy_input, reward, action, policy.nb_actions)

        if prev_reward != reward:
            data_holder.log_summary()
            prev_reward = reward

        if done:  # When Game env say it's done - end of episode.
            LOGGER.info("{%s} Episode done! Reward sum: %.2f , Frames: %d",
                        data_holder.episode_number,
                        data_holder.rewards_episode.sum(),
                        data_holder.record_counter_episode)

            observation = env.reset()
            prev_observation = None
            data_holder.next_episode()

            if data_holder.record_counter >= batch_size and not ARGS.test:
                LOGGER.info("Update weights from %d frames, average rewards: %.2f",
                            data_holder.record_counter,
                            data_holder.rewards.sum() / data_holder.episode_number_batch)

                with summary_writer.as_default():
                    policy.train_step(
                        data_holder.observations,
                        np.vstack(data_holder.labels),
                        np.vstack(data_holder.rewards_discounted),
                        step=tf.constant(data_holder.episode_number, dtype=tf.int64)
                    )

                data_holder.next_batch()
            elif data_holder.record_counter >= batch_size:
                data_holder.next_batch()

        if ARGS.test:
            time.sleep(0.01)

        if data_holder.episode_number > ARGS.episodes:
            raise TrainingStoppedException('Training finished.')


def main():
    """Run the main pipeline."""

    summary_writer, chk_path = setup_outputs()
    policy = Policy(nb_actions=6, learing_rate=ARGS.learning_rate)

    try:
        policy.load(chk_path)
    except tf.errors.NotFoundError:
        LOGGER.warning("Checkpoint Not Found: %s", chk_path)

    try:
        learning(ARGS.env, policy, ARGS.batch_size, summary_writer)
    except (KeyboardInterrupt, TrainingStoppedException):
        # Stop learning save files
        if not ARGS.test:
            policy.save(chk_path)
            with tf.io.gfile.GFile(join(dirname(chk_path), 'step.txt'), 'w') as txt_file:
                txt_file.write(str(tf.summary.experimental.get_step()))


if __name__ == "__main__":
    LOGGER = get_logger('gym')
    LOGGER.setLevel(logging.DEBUG)

    ARGS = user_args()

    if ARGS.gpu:
        PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

        main()
    else:
        with tf.device('cpu:0'):
            main()
