# Policy Gradient for image based robotics tasks

Implementation of Policy Gradient method, reinforcement learning algorithm to solve image based robotics problem. As of now it's clear that this is not the most efficient way to solve it. But before jumping to implementations such as [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542) it's beneficial to understand more simpler algorithms and their upsides and downsides.

This is 1. Experiment, every next experiment will be more sophisticated and hopefully with better results.

## Features

1. Implemented PG from scratch in simple way
2. Tensorflow 2.x support
3. Visuals in Tensorboard
4. Ready to be run on Cloud (Google Cloud Platform)

## Requirements

1. Python 3.7+, Tensorflow](https://www.tensorflow.org/) >=2.1, Linux (Debian)
2. My [custom Gym environment](https://github.com/Naurislv/image_based_fetch_gym_env.git)
3. Install all dependecies: `pip install -r requirements.txt`

## Training locally

1. `git clone https://github.com/Naurislv/1_experiment_pg_robotics.git`
2. `cd 1_experiment_pg_robotics/policy_gradient_robot_learning`
3. GPU: `python learning.py --episodes 17000 --batch_size 10000`
4. CPU: `export CUDA_VISIBLE_DEVICES= && python learning.py --gpu False --episodes 17000 --batch_size 10000`

## Training on cloud

1. Read [gcp reinforcement learning tutorial](https://cloud.google.com/blog/products/ai-machine-learning/deep-reinforcement-learning-on-gcp-using-hyperparameters-and-cloud-ml-engine-to-best-openai-gym-games)
2. Run `bash run_gcloud.bash`

# OpenAI Gym test results

Plots are generated using [this code](./PlayGround.ipynb).

## Pong-v4 with Policy Gradient

### Policy NN : [1 hidden layer fully connected network](./Nets/KarpathyNet.py)

I trained this network for ~18k Episodes for each test because it is very time consuming - around 17h.

#### 1. Test

For this Test specific Pong image preprocessing were used to converge faster :

```
img[img == 17] = 0 # erase background (background type 1)
img[img == 192] = 0 # erase background (background type 2)
img[img == 136] = 0 # erase background (background type 3)
img[img != 0] = 1 # everything else (paddles, ball) just set to 1

img = img[17:96, :]
```

In future this and similar preprocessing will be replaced/removed so this algorithm might be used for different games. It's also worth to note that if we remove preprocessing - learning then with this network is much slower and it's possible that it wont peak this high. See 2. Test.

Hyperparameters:

* RMSPropOptimizer(learning_rate=0.001, decay=0.99)
* Discounted Rewards gamma = 0.99
* Batch size (number of episodes) = 10

![alt text][image2]

#### 2. Test

Without preprocessing (without removing background and binarization). We can see that learning is much slower however there is no sign that it won't converge to same level as in 1. Test eventually.

Hyperparameters:

* RMSPropOptimizer(learning_rate=0.001, decay=0.99)
* Discounted Rewards gamma = 0.99
* Batch size (number of episodes) = 10

![alt text][image3]

## Known issues

* When batches are big 6+ then updates needs to be done with 7k+ images which causes GPU Out Of Memory.
* Pongs (tested v0, v4) first frame from env.reset() returns different frame (different colors) than env.step() therefor 'recording' starts only from 3rd frame
* Currently works only with discrete action space

## Motivation

I am working on sophisticated robotics solutions for Smart Manufacturing use-cases. In my experience I have obsereved that more simpler solutions often works better in production than more sophisticated, think OpenCV vs Machine Learning. Supervised Machine Learning methods are very hard to implement in factory because you almost never have enough nor right data to feed for agorithms. While solutions implementing classical Computer Vision algorithms (OpenCV) works well in many cases, they introduce other challenges such as calibration requirements, longer robot setups, more complex development, engineering piplines etc.

Note that I have specific robotics use-cases in mind which I would like to replicate in simulator therefore just any simulator with or without robot will not work and this is why I am motivated to create something working for my specific needs and not the approach which would work with multiple environments.

Reinforcement Learning is approach which is changing the game ([Osaro](https://www.osaro.com/), [Vicarious](https://www.vicarious.com/)). Instead of jumping to latest and greatest I have decided to start with simplest and build my knowledge and experementation experience from here.

[//]: # (Image References)

[image1]: ./images/atari_game_performance.png "Atari game performance compare"
[image2]: ./images/pong_pg_results.png "Pong Policy Gradient Results"
[image3]: ./images/pong_pg_results_without_preprocessing.png "Pong Policy Gradient Results without Preprocessing"

## Atari game performance comparison between algorithms and Human

![alt text][image1]