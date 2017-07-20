# OpenAI Gym RL algorithm implementation

## Dependecies

* pip install gym
* pip install imageio
* sudo apt-get install swig
* pip install box2d

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