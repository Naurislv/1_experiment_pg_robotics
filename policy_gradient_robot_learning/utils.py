"""Utilities used throught project."""

# Standard imports
import logging

# Dependency imports
import numpy as np
# import tensorflow as tf


def get_logger(logger_name):
    """Get logger with predefined settings."""

    logging.basicConfig(level=logging.DEBUG, format='[%(name)s] %(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger(logger_name)

    return logger


LOGGER = get_logger('utils')


def pong_img_preproc(prev_img, img):
    """Preprocess 210x160x3 uint8 frame for PingPong-v4 Gym Environment."""

    # downsample by factor of 2
    img = img[::2, ::2]  # Downsample but keep all channels

    img[img == 17] = 0 # erase background (background type 1)
    img[img == 192] = 0 # erase background (background type 2)
    img[img == 136] = 0 # erase background (background type 3)
    img[img != 0] = 255 # everything else (paddles, ball) just set to 1
    img = img[17:96, :]

    # Insert motion in frame by subtracting previous frame from current
    if prev_img is not None:
        policy_input = img - prev_img
    else:
        policy_input = np.zeros_like(img)

    prev_img = img

    return prev_img, policy_input


# @tf.function(experimental_relax_shapes=True)
# def _pong_img_preproc(prev_img, img):
#     """Preprocess 210x160x3 uint8 frame for PingPong-v4 Gym Environment."""

#     # downsample by factor of 2
#     img = tf.image.resize(img, (img.shape[-3], img.shape[-2]))

#     img[img == 17] = 0 # erase background (background type 1)
#     img[img == 192] = 0 # erase background (background type 2)
#     img[img == 136] = 0 # erase background (background type 3)
#     img[img != 0] = 255 # everything else (paddles, ball) just set to 1
#     img = img[17:96, :]

#     # Insert motion in frame by subtracting previous frame from current
#     if prev_img is not None:
#         policy_input = tf.math.subtract(img, prev_img)
#     else:
#         policy_input = tf.zeros_like(img)

#     prev_img = img

#     return prev_img, policy_input


def discount_rewards(reward_his, gamma=.99):
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
