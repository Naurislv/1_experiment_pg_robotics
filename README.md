# Policy Gradient for image based robotics tasks

Implementation of Policy Gradient method, reinforcement learning algorithm to solve image based robotics problem. As of now it's clear that this is not the most efficient way to solve it. But before jumping to implementations such as [Asymmetric Actor Critic for Image-Based Robot Learning](https://arxiv.org/abs/1710.06542) it's beneficial to understand more simpler algorithms and their upsides and downsides.

This is 1. Experiment, every next experiment will be more sophisticated and hopefully with better results.

## Motivation

I am working on sophisticated robotics solutions for Smart Manufacturing use-cases. In my experience I have obsereved that more simpler solutions often works better in production than more sophisticated, think OpenCV vs Machine Learning. Supervised Machine Learning methods are very hard to implement in factory because you almost never have enough nor right data to feed for agorithms. While solutions implementing classical Computer Vision algorithms (OpenCV) works well in many cases, they introduce other challenges such as calibration requirements, longer robot setups, more complex development, engineering piplines etc.

Reinforcement Learning is approach which is changing the game ([Osaro](https://www.osaro.com/), [Vicarious](https://www.vicarious.com/)). Instead of jumping to latest and greatest I have decided to start with simplest and build my knowledge and experementation experience from here.

[//]: # (Image References)

[image1]: ./Images/atari_game_performance.png "Atari game performance compare"
[image2]: ./Images/pong_pg_results.png "Pong Policy Gradient Results"
[image3]: ./Images/pong_pg_results_without_preprocessing.png "Pong Policy Gradient Results without Preprocessing"

## Requirements

Tensorflow 2.0+

Install all dependecies: `pip install -r requirements.txt`

## Known issues

* Current code does run faster on CPU (Intel(R) Core(TM) i7-6800K) than on GPU (GTX-1080) even though total CPU load is around 20% (While training 100%, thanks to TF multiprocessing). This may be because of latency between GPU and CPU

* Pongs (tested v0, v4) first frame from env.reset() returns different frame (different colors) than env.step() therefor 'recording' starts only from 3rd frame

* Currently works only with discrete action space

## Training

`python OpenAIGym.py --render False --gpu False --name Test`

Sample output of training process

```
[2017-07-30 14:23:53,534] 13270. [52363.94s] FPS: 1093.26, Reward Sum: -2.0
[2017-07-30 14:23:53,583] 13270. [52363.99s] FPS: 1025.90, Reward Sum: -3.0
[2017-07-30 14:23:53,631] 13270. [52364.04s] FPS: 1040.43, Reward Sum: -3.0
[2017-07-30 14:23:53,648]
[2017-07-30 14:23:53,648] Episode done! Reward sum: -4.00 , Frames: 6369
[2017-07-30 14:23:53,648]
[2017-07-30 14:23:53,793] Update weights from 64278 frames with average score: 4.6
[2017-07-30 14:23:53,793] Used action space: {0: 416, 1: 946, 2: 22232, 3: 32888, 4: 6650, 5: 1146}
[2017-07-30 14:23:55,945] 13271. [52366.35s] FPS: 937.53, Reward Sum: 0.0
```

It is easy to switch between different Neural Network architectures because all of them return necesarry Tensorflow objects for Policy Gradient to use. Just comment current neural network import and uncomment other in [PolicyGradient.py](./PolicyGradient.py). For example we will use KarpathyNet.py network:

```
# from Nets import MaxoutNet as policy_net
# from Nets import GuntisNet as policy_net
from Nets import KarpathyNet as policy_net
```

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

### Policy NN : [2 x conv, 2 x fully connected ](./Nets/GuntisNet.py)

This network was only trained with input images as human would see, using all 3 color channels and no cropping, except it's downsampled (resized without interpolation) two times.

## Extra Reasearch [2017.18.07]

### Virtual Environments for Self Driving Cars

There are few virtual environments specifically for self driving cars at the moment. There of-course are more than here - Google just didn't showed them to me. Also they all does not covery full spectrum of real self driving cars but in each you can find something you may be interested in.

* [OpenAI Gym](https://gym.openai.com/) - Specially created to train Agents for RL. There are also environments which is more suited for self driving car altough nothing really close to reality but to start coding RL - this is really great place.

* [OpenAI Universe](https://gym.openai.com/) - There are plenty of more sophisticated games than can be found in OpenAI Gym. Great thing about Universe is that you can actually play all games by yourself before to start work with them. It's harder to set everything up and running but totally worth it. Seeo [GitHub](https://github.com/openai/universe) for more information. Probably the best environment to test self driving car algorithms is [GTA-V](https://universe.openai.com/envs/gtav.SaneDriving-v0) which currently is in *coming soon* state. If you are RL beginner then you really should start with OpenAI Gym.

* [Udacity Self Driving Car simulator](https://github.com/udacity/self-driving-car-sim) - Great for self driving - behavioral cloning. But not great for RL algorithms when you need to run more than one agent and get metrics of progress. Also there are no other cars and only few tracks.

* [Udacity Self Driving Car Term 2 simulators](https://github.com/udacity/self-driving-car-sim/releases) Multiple simulators for multiple puropses. See [GitHub](https://github.com/udacity/self-driving-car-sim) for more information.

* [Self Driving Cars in Browser](http://janhuenermann.com/projects/learning-to-drive) is virtual environment written in JavaScript. Made to train RL algorithms.

* [MIT Self Driving Cars](http://selfdrivingcars.mit.edu/) is Home Page specifically dedicated to Self Driving Cars. And there you can find [DeepTesla](http://selfdrivingcars.mit.edu/deeptesla/) and [DeepTraffic](http://selfdrivingcars.mit.edu/deeptraffic/) virtual environments also developed in browser. Check [LeaderBoard](http://selfdrivingcars.mit.edu/leaderboard/) for DeepTraffic.

* [TORCS](https://en.wikipedia.org/wiki/TORCS) is the open racing car simulator. I wasn't able to find decent home page for this simmulator. [Download](https://sourceforge.net/projects/torcs/) and more [info](http://torcs.sourceforge.net/). Also worth checking [publication](http://personal.ee.surrey.ac.uk/Personal/N.Pugeault/projects/RRUDZITS_LEARNING_AUTONOMOUS_DRIVING_FINAL.pdf) by Reinis Rudzītis.

## Atari game performance comparison between algorithms and Human

![alt text][image1]