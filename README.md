# drlnd-p1-nav
# Project details
The aim of this repository is to provide a solution to Udacity's [DRLND project 1.](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation) The challenge of this programm is to create an agent which is able to act in a modified version of the Unity bananas environement. Within this environement an agent needs to collect yellow bananas by running over it. Every yellow banana yields a reward of +1 for the agent. Unfortunately, the environment contains also blue bananas. Running over a blue banana results in a negative reward of -1. The environment is considered solved when the agents earns an average total reward of +13 or higher for 100 consecutives episodes.

At every time step, the agent has the option to choose on of the following actions,
* move forward,
* move backward,
* turn left,
* turn right,
resulting in action space of size 4. The state space has of 37 dimensions including the velocity of the agent and ray-based perceptions of the objects in front of the agent.

The code of this agent is based on the [DRLND dqn exercise].(https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn). The repository contains the following files.
* DQAgent.py - *Contains the core of the agent*
* train_agent.py - *Contains
* banana_agent.py -

Additionally an report which describes the learning algorithm in more detail and which evaluates the performance of the agent can be found here.

# Getting started
The agent itself requires only standard python3, PyTorch and Numpy. 
1. Install [Anaconda](https://www.anaconda.com/download)
2. Clone [Udacitys'DRLND repository(https://github.com/udacity/deep-reinforcement-learning)] and follow the installation instructions.
3. Download the corresponding Unity environment for your system as described [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started)
4. Unzip the environment and change the *PATH_TO_ENV* in variable line 1 in banana_agent.py accordingly.

# Instructions
You have two options:
1. You can run the agent using the provided pre-trained weights and observe how the agents performs over 100 episodes by executing the following command:
*python banana_agent.py*
2. Alternatively you can retrain the agent yourself. Please note that this overwrites the provided pre-trained weights if you do not modify the code before.
*python banana_agent.py retrain*
The last option is especially interesting if you want to play with the agent's parameters yourself.

# More advanced stuff
By using banana_agent.py as a template, you can easily adapt the agent to interact within other Unity environments. To use the agent with other environement frameworks (e.g. Open AI Gym) you have to modify the *train_agent* function of train_agent.py first.
