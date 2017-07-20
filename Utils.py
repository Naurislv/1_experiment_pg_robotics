"""Utilities used throught project."""

# Standard imports
import logging
import os
import time

# Other Imports
import scipy.misc

def save_im(file_name, img, overwrite=False, create_subs=False):
    """Save image file to local system.

    If file exist rename.
    """
    logging.debug('Saving image: %s', file_name)

    if not os.path.exists(os.path.dirname(file_name)) and create_subs:
        os.makedirs(os.path.dirname(file_name))

    if os.path.exists(file_name) and overwrite is False:
        file_name = file_name.split('.')
        extension = file_name[-1]
        file_name = '.'.join(file_name[0:-1])
        file_name = file_name + '_' + str(int(round(time.time() * 1000))) + '.' + extension

    scipy.misc.imsave(file_name, img)
