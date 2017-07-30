# OpenAI Gym RL algorithm implementation

The main goal is to train self driving cars to learn to drive by themselves but while learning to do so I'm starting with basics like OpenAI Gym Pong game and so on. So in this repository I will cover multiple OpenAI Gym games but will focus on self driving cars while implementing.

[//]: # (Image References)

[image1]: ./Images/atari_game_performance.png "Atari game performance compare"
[image2]: ./Images/pong_pg_results.png "Pong Policy Gradient Results"

## Dependecies

* pip install gym
* pip install imageio
* sudo apt-get install swig
* pip install box2d
* pip install gym[atari]
* pip install tensorflow

## Issues

* Current code does run faster on CPU (Intel(R) Core(TM) i7-6800K) than on GPU (GTX-1080) even though total CPU load is around 20% (While training 100%, thanks to TF multiprocessing). This may be because of latency between GPU and CPU

* Pongs (tested v0, v4) first frame from env.reset() returns different frame (different colors) than env.step() therefor 'recording' starts only from 3rd frame

* Code need to be refactorized

* Currently works only with discrete action space

## Atari game performance comparison between algorithms and Human

![alt text][image1]

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

It is easy to switch between different Neural Network architectures because all of them return necesarry Tensorflow objects for Policy Gradient to use. Just comment current neural network import and uncomment other in [PolicyGradient.py](./PolicyGradien.py). For example we will use KarpathyNet.py network:

```
# from Nets import MaxoutNet as policy_net
# from Nets import GuntisNet as policy_net
from Nets import KarpathyNet as policy_net
```

# OpenAI Gym test results

Plots are generated using [this code](./PlayGround.ipynb).

## Pong-v4 with Policy Gradient

Testing results using [1 hidden layer fully connected network](./Nets/KarpathyNet.py) as policy Network.

Note that for this Test - specific Pong image preprocessing were used :

```
img[img == 17] = 0 # erase background (background type 1)
img[img == 192] = 0 # erase background (background type 2)
img[img == 136] = 0 # erase background (background type 3)
img[img != 0] = 1 # everything else (paddles, ball) just set to 1

img = img[17:96, :]
```

In future this and similar preprocessing will be replaced/remove so this algorithm might be used for different games. It's also worth to note that if we remove preprocessing - learning then with this network is much slower and it's possible that it wont peak this high.

Hyperparameters:

* RMSPropOptimizer(learning_rate=0.001, decay=0.99)
* Discounted Rewards gamma = 0.99
* Batch size (number of episodes) = 10

![alt text][image2]

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

### Reinforcement Learning for Self Driving Cars

There is no specific RL algorithm made just for Self Driving Cars of-course so here I will also cover RL algorithms in general and may add some sources and comments which specifically relates to self driving cars. You may solve some problems which may just work for Self Driving cars and not other domains by editing RL algorithm. There are plenty of materials online so I will try to conver only those which are more relevant to our topic.

* Probably the best starting point for RL is [Deep Reinforcement Learning: Pong from pixels](http://karpathy.github.io/2016/05/31/rl/) by Karpathy. Where he give sources for other materials for letcures. Explains why __Policy Gradient__ is better than __Deep Q__, explains why it works and actually show how to implement it purely in using Python and numpy.

* Also really great [guide](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) for beginners by [Arthur Juliani](https://medium.com/@awjuliani)

* [A3C](https://arxiv.org/abs/1602.01783). We propose a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. We present asynchronous variants of four standard reinforcement learning algorithms and show that parallel actor-learners have a stabilizing effect on training allowing all four methods to successfully train neural network controllers. The best performing method, an asynchronous variant of actor-critic, surpasses the current state-of-the-art on the Atari domain while training for half the time on a single multi-core CPU instead of a GPU. Furthermore, we show that asynchronous actor-critic succeeds on a wide variety of continuous motor control problems as well as on a new task of navigating random 3D mazes using a visual input.

* [Safe, Multi-Agent, Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/1610.03295v1). Autonomous driving is a multi-agent setting where the host vehicle must apply sophisticated negotiation skills with other road users when overtaking, giving way, merging, taking left and right turns and while pushing ahead in unstructured urban roadways. Since there are many possible scenarios, manually tackling all possible cases will likely yield a too simplistic policy. Moreover, one must balance between unexpected behavior of other drivers/pedestrians and at the same time not to be too defensive so that normal traffic flow is maintained.

* [Deep Reinforcement Learning framework for Autonomous Driving](https://arxiv.org/pdf/1704.02532.pdf). Reinforcement  learning  is  considered  to  be  a  strong  AI paradigm which can be used to teach machines through interaction with the environment and learning from their mistakes. Despite its perceived utility, it has not yet been successfully appliedin automotive applications.

## Credits

* Guntis Bārzdiņš
* Renārs Liepiņš