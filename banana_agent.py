PATH_TO_ENV="/home/marco/Dropbox/Data_Science/Udacity/DRLND/deep-reinforcement-learning/p1_navigation/Banana_Linux_NoVis/Banana.x86_64"

from unityagents import UnityEnvironment
import numpy as np
import torch
from DQAgent import DQAgent
from train_agent import train_agent, test_agent
from collections import deque
import sys
from matplotlib import pyplot as plt

if __name__=="__main__":

    #Initialize Environment
    print("Loading environment from", PATH_TO_ENV)
    env = UnityEnvironment(file_name=PATH_TO_ENV)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # get dimensions of action space and state space
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    # Init agent
    agent = DQAgent(state_size=state_size, action_size=action_size, hidden_layers = [64, 64],double_ql = True)

    if len(sys.argv)>=2 and sys.argv[1]=="retrain":
        print("Retraining agent")
        scores = train_agent(agent, env, brain_name, max_score = 16.1)
        np.save("train_scores.npy",np.array(scores))

    else: # Run the agent with pretrained weights
        print("Running agent with pretrained weights")

        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load('banana_weights.pth'))
        scores = test_agent(agent, env, brain_name)
        np.save("test_scores.npy",np.array(scores))
        # Create Plot of Scores
        plt.figure()
        plt.hist(scores, 13)
        plt.title("Histogram of scores during 100 episodes")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig("agent_test_histogram.png")
        plt.figure()
        plt.plot(np.arange(len(scores)),scores)
        plt.hlines(13.,0, 100, "r")
        plt.title("Histogram of scores during 100 episodes")
        plt.ylabel("Score")
        plt.xlabel("Episode")
        plt.savefig("agent_test_performance.png")
        if np.mean(scores)>13.:
            print("Test successfull. Average score greater than 13.")
        else:
            print("Test failed. Average score smaller than 13.")
    env.close()
    
