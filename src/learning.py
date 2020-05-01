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
from os.path import dirname, join, exists

# Dependency imports
import gym
import tensorflow as tf
import numpy as np

# Local imports
from policy_gradient import Policy
from argument_parser import user_args
from utils import get_logger, pong_img_preproc
from data_management import DataManager


def setup_outputs():
    """Setup Tensorboard and runtime outputs directories and files"""

    prefix = ''
    if ARGS.test:
        prefix = 'test'

    session_descriptor = (f"{prefix}lr[{ARGS.learning_rate}]_bs[{ARGS.batch_size}]_"
                          f"policy[{ARGS.policy}]_model[{ARGS.model}]_env[{ARGS.env}]")

    session_id = ARGS.session_id
    if not session_id:
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    train_log_dir = f'outputs/{session_id}/{session_descriptor}/'
    summary_writer = tf.summary.create_file_writer(train_log_dir)

    chk_path = f'outputs/{session_id}/model'

    step_path = f'outputs/{session_id}/{prefix}step.txt'
    if exists(step_path):
        with open(step_path, "r") as txt_file:
            episode_number = int(txt_file.read())

        tf.summary.experimental.set_step(episode_number)

    return summary_writer, chk_path


def make_env(name):
    """Create an environment."""

    env = gym.make(name)
    observation = env.reset()
    LOGGER.info("%s initialized", name)

    for _ in range(1):
        observation, _, _, _ = env.step(0)

    return env, observation


def learn(env_name, policy, batch_size, summary_writer):
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

        action = policy.sample_action(policy_input)[0][0]

        # step the environment and get new measurements
        observation, reward, done, _ = env.step(action)

        data_holder.record_data(policy_input, reward, action, policy.nb_actions)

        if prev_reward != reward:
            data_holder.log_summary()
            prev_reward = reward

        if done:  # When Game env say it's done - end of episode.
            LOGGER.info("Episode done! Reward sum: %.2f , Frames: %d",
                        data_holder.rewards_episode.sum(), data_holder.record_counter_episode)

            observation = env.reset()
            prev_observation = None
            data_holder.next_episode()

            if (data_holder.episode_number - 1) % batch_size == 0:
                LOGGER.info("Update weights from %d frames with average score: %s",
                            data_holder.record_counter, data_holder.rewards.sum() / batch_size)

                with tf.device('cpu:0'):
                    with summary_writer.as_default():
                        policy.train_step(
                            data_holder.observations,
                            np.vstack(data_holder.labels),
                            np.vstack(data_holder.rewards_discounted),
                            step=tf.constant(data_holder.episode_number - 1, dtype=tf.int64)
                        )

                data_holder.next_batch()


def main():
    """Run the main pipeline."""

    summary_writer, chk_path = setup_outputs()
    policy = Policy(nb_actions=6, learing_rate=ARGS.learning_rate)

    try:
        policy.load(chk_path)
    except tf.errors.NotFoundError:
        LOGGER.warning("Checkpoint Not Found: %s", chk_path)

    try:
        learn(ARGS.env, policy, ARGS.batch_size, summary_writer)
    except KeyboardInterrupt:  # Stop learning by "CTRL + C" and save files
        if not ARGS.test:
            policy.save(chk_path)
            with open(join(dirname(chk_path), 'step.txt'), "w") as txt_file:
                txt_file.write(str(tf.summary.experimental.get_step()))


if __name__ == "__main__":
    PHYSICAL_DEVICES = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(PHYSICAL_DEVICES[0], True)

    LOGGER = get_logger('gym')
    LOGGER.setLevel(logging.DEBUG)

    ARGS = user_args()

    main()
