"""Provide user input functionality."""

# Standard imports
import argparse

def str2bool(val):
    """String to boolean type."""

    if isinstance(val, bool):
        return val
    if val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def user_args():
    """Parse user input arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_small', type=str2bool, default=False,
                        help="Whether to record and save video or not. Boolean.")
    parser.add_argument('--video_big', type=str2bool, default=False,
                        help="Whether to record and save video or not. Gym built "
                             "in command. Also create report .json files. Boolean.")
    parser.add_argument('--render', type=str2bool, default=False,
                        help="Whether to render video for visual or not.")
    parser.add_argument('--gpu', type=str2bool, default=False, help="Use GPU.")
    parser.add_argument('--sessiod_id', type=str, default='',
                        help="Session ID, will be used to save and load checkpoint and statistic "
                             "files.")

    args = parser.parse_args()

    return args
