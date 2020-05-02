"""Install policy gradient robot learning as package.

Requirement to run as a job on Google Cloud Platform.
"""

from setuptools import setup

setup(
    name='policy_gradient_robot_learning',
    version='0.1',
    install_requires=['gym']
)
