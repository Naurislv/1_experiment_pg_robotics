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

    parser.add_argument('--test', type=str2bool, default=False,
                        help="Whether to test trained model or not.")
    parser.add_argument('--session_id', type=str, default='',
                        help="Session ID, will be used to save and load checkpoint and statistic "
                             "files.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--env', type=str, default='Pong-v4', help="Environment name")
    parser.add_argument('--model', type=str, default='NatureCNN', help="Policy network name")
    parser.add_argument('--policy', type=str, default='PG', help="Policy algorithm name")

    args = parser.parse_args()

    return args
