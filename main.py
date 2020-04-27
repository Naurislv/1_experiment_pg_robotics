# Standard imports
import time
import logging
import os

# Dependency imports
import gym
import image_based_robot_env
import tensorflow as tf


def make_env():
    """Make Gym environment"""
    env = gym.make("image-based-robot-env-v0")

    # These steps are required because of the workaround of different bugs experience during testing
    obs = env.reset()
    env.render('rgb_array')
    _, _, _, _ = env.step([0, 0, 0, 0])

    return env


if __name__ == "__main__":

    ENV = make_env()
