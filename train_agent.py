import numpy as np
import torch
from DQAgent import DQAgent
from collections import deque


def train_agent(agent, env, brain_name, save_name = 'banana_weights.pth',
                N_episodes = 2000, max_t=10000000, max_score=2000.,
                scores_intervall = 100,
                eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995):
    """Trains a DQAgent within an Unity Environment.
    Finishes when a specified score is achieved or when the numter of training episodes is exceeded
        
    Params
    =====
        agent (DQAgent): agent to be trained
        env (): environment
        brain_name (): agent to be trained
        save_name (str): name of the file where the weights are stored after training
        N_episodes (int): Max number of training episodes
        max_t (int): Max number of time steps within each episode before the episode is interrupted
        max_score (float): If the agent achieves an average score over the last score_intervall episodes
                           greater than max_score, training is considered successful and the algorithm terminates
        scores_interavall (int): Number of episodes for which the average score is calculated
        eps_start (float): start value for epsilon greedy policy
        eps_end (float): end value for epsilon greedy policy
        eps_decay (float): decay rate of epsilon greedy policy


    Returns
    ======
        scores ([floats]): list containing the score achieved during every episode of the training

    """

    scores_window = deque(maxlen=scores_intervall)
    scores = []
    eps = eps_start
    
    # Iterate through episodes
    for n in range(N_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score
        eps = max(eps_end, eps_decay*eps)

        # iterate through time steps
        for t in range(max_t):
            action = agent.act(state, eps)                 # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores.append(score)
        scores_window.append(score)

        # Output current average score every scores_window episodes
        if np.mod(n,scores_intervall)==0 and n>0:
            print("Average Score: {}".format(np.mean(scores_window)),
                 "Min: {}".format(np.min(scores_window)),
                 "Max: {}".format(np.max(scores_window)),
                 "Std: {}".format(np.std(scores_window)),)

        # Check if environment is considered solved
        if (np.mean(scores_window)>max_score):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(n-scores_intervall, np.mean(scores_window)))
            break

    # Save the weights and return the score
    torch.save(agent.qnetwork_local.state_dict(), save_name)
    return scores

def test_agent(agent, env, brain_name, N_episodes = 100, max_t=10000000):
    """Tests a DQAgent within an Unity Environment.
            
    Params
    =====
        agent (DQAgent): agent to be trained
        env (): environment
        brain_name (): agent to be trained
        N_episodes (int): Max number of training episodes
        max_t (int): Max number of time steps within each episode before the episode is interrupte


    Returns
    ======
        scores ([floats]): list containing the score achieved during every episode of the test

    """

    scores = []
    
    # Iterate through episodes
    for n in range(N_episodes):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0                                          # initialize the score

        # iterate through time steps
        for t in range(max_t):
            action = agent.act(state, 0.)                  # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            score += reward                                # update the score
            state = next_state                             # roll over the state to next time step
            if done:                                       # exit loop if episode finished
                break

        scores.append(score)

    print("Average Score: {}".format(np.mean(scores)),
          "Min: {}".format(np.min(scores)),
          "Max: {}".format(np.max(scores)),
          "Std: {}".format(np.std(scores)),)
    return scores


# opti paras: double q on
# 64 64 , max score 16.5
