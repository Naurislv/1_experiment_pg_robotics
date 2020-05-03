"""Install policy gradient robot learning as package.

Requirement to run as a job on Google Cloud Platform.
Inspired from:
    https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/blogs/rl-on-gcp/
    DQN_Breakout/rl_on_gcp/setup.py
"""

import subprocess
from distutils.command.build import build as _build

from setuptools import setup
from setuptools import find_packages
import setuptools


class build(_build):
    """A build command class that will be invoked during package install.
        The package built using the current setup.py will be staged and later
        installed in the worker using `pip install package'. This class will be
        instantiated during install for this specific scenario and will trigger
        running the custom commands specified.
        """
    sub_commands = _build.sub_commands + [('CustomCommands', None)]


# The list of required libraries is taken from:
# https://github.com/openai/gym#installing-everything
_LIBS = ('cmake xvfb xorg-dev python3-opengl libboost-all-dev libsdl2-dev swig').split()


CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', 'install', '-y'] + _LIBS,
]


class CustomCommands(setuptools.Command):
    """A setuptools Command class able to run arbitrary commands."""

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print('Running command: %s' % command_list)
        ppn = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = ppn.communicate()
        print('Command output: %s' % stdout_data)
        if ppn.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (command_list, ppn.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)

REQUIRED_PACKAGES = [
    "gym[atari]>=0.17.1"
]

setup(
    name='policy_gradient_robot_learning',
    version='0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Policy Gradient for Robot learning',
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    }
)
