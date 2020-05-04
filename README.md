[//]: # (Image References)

[image1]: ./images/atari_game_performance.png "Atari game performance compare"

# Policy Gradient for image based robotics tasks

Implementation of Policy Gradient method, reinforcement learning algorithm to solve image based robotics problem. As of now it's clear that this is not the most efficient way to solve it. But before jumping to implementations such as [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542) it's beneficial to understand more simpler algorithms and their upsides and downsides.

This is 1. Experiment, every next experiment will be more sophisticated and hopefully with better results.

## Table of contents

- [Features](#features)
- [Requirements](#requirements)
- [Training locally](#training-locally)
- [Training on cloud](#training-on-cloud)
- [Known issues](#known-issues)
- [Debugging GCP](#debugging-gcp)
- [Environment image preprocessing](#environment-image-preprocessing)
  * [1. Image based fetch environment](#1-image-based-fetch-environment)
  * [2. Pong-v4](#2-pong-v4)
- [Motivation](#motivation)
- [Atari game performance comparison between algorithms and Human](#atari-game-performance-comparison-between-algorithms-and-human)

## Features

1. Implemented PG from scratch in simple way
2. Tensorflow 2.x support
3. Visuals in Tensorboard
4. Ready to be run on Cloud (Google Cloud Platform)

## Requirements

1. Python 3.7+,
2. [Tensorflow](https://www.tensorflow.org/) >=2.1
3. Linux (Debian), tested on Ubuntu 18.04
4. __[skip]__ Install [image_based_fetch_gym_env](https://github.com/Naurislv/image_based_fetch_gym_env.git)
5. Install all dependecies: `pip install -r requirements.txt`

## Training locally

1. `git clone https://github.com/Naurislv/1_experiment_pg_robotics.git`
2. `cd 1_experiment_pg_robotics/policy_gradient_robot_learning`
3. GPU: `python learning.py --episodes 17000 --batch_size 10000`
4. CPU: `export CUDA_VISIBLE_DEVICES= && python learning.py --gpu False --episodes 17000 --batch_size 10000`

## Training on cloud

For training I am using [GCP](https://cloud.google.com/) (Google Cloud Platform). If you are not familiar with Google AI Platform then read [gcp reinforcement learning tutorial](https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games). It is outdate in terms of GCP AI Platform syntax but principles are still up to date.

1. Make sure that right parameters are set in [run_gcloud.bash](run_gcloud.bash) (by default it should work well)
2. Make sure to use right configuration in [hyperparam.yaml](hyperparam.yaml) (by default it should work well)
3. Execute `bash run_gcloud.bash`
4. Open GCP Console and run to see live metrics: `tensorboard --logdir=gs://robot_learning/policy_gradient_robot_learning`

## Known issues

- When batches are big 10k+ images which causes GPU run out of memory.
- Pong-(v0, v4) first frame from `env.reset()` returns different frame (different colors) than `env.step()` therefor recording starts only from first `.step()` frame instead of `.reset()` frame.
- Currently works only with discrete action space


## Debugging GCP

### Run GCP command locally

Execute `bash run_gcloud local`

### Debug package

To quickly check if everything is working one can spin up Ubuntu 18.04 Docker image and follow these steps:

1. Copy or clone repository.
2. Build the package (install apt-get dependencies): `python3 setup.py build`
3. Install package and dependencies: `python3 setup.py install`
4. Run package: `python3 -m policy_gradient_robot_learning.main --gpu False`


## Environment image preprocessing

For faster convergence I use some image preprocessing.

### 1. Image based fetch environment

_Currently not supported. Environment accessible [here](https://github.com/Naurislv/image_based_fetch_gym_env.git)_

### 2. Pong-v4

For testing purposes I used image preprocessing for algorithm to converge faster:

```python
img[img == 17] = 0 # erase background (background type 1)
img[img == 192] = 0 # erase background (background type 2)
img[img == 136] = 0 # erase background (background type 3)
img[img != 0] = 1 # everything else (paddles, ball) just set to 1

img = img[17:96, :]
```

## Motivation

I am working on sophisticated robotics solutions for Smart Manufacturing use-cases. In my experience I have obsereved that more simpler solutions often works better in production than more sophisticated, think OpenCV vs Machine Learning. Supervised Machine Learning methods are very hard to implement in factory because you almost never have enough nor right data to feed for agorithms. While solutions implementing classical Computer Vision algorithms (OpenCV) works well in many cases, they introduce other challenges such as calibration requirements, longer robot setups, more complex development, engineering piplines etc.

Note that I have specific robotics use-cases in mind which I would like to replicate in simulator therefore just any simulator with or without robot will not work and this is why I am motivated to create something working for my specific needs and not the approach which would work with multiple environments.

Reinforcement Learning is approach which is changing the game ([Osaro](https://www.osaro.com/), [Vicarious](https://www.vicarious.com/)). Instead of jumping to latest and greatest I have decided to start with simplest and build my knowledge and experementation experience from here.

This is something what GCP does when you send the job to cloud. I used "[training-data-analyst](https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/rl-on-gcp/DQN_Breakout)" repository as an example.

## Atari game performance comparison between algorithms and Human

![alt text][image1]
