import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0., double_ql=False, prio_replay=False, p_alpha = 1.,
                 hidden_layers = [128,128], buffer_size = 100000, batch_size = 64, gamma = 0.99,
                 tau = 1e-3, lr = 5e-4, update_every = 4) :
        """Initializes an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            double_ql (bool): turns double ql-learning on or off
            prio_replay (bool): uses a prioritized replay buffer
            p_alpha (float): parameter to shape the distribution of the prioritized weights
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target network parameters
            lr (float): learning rate of online q-network gradient descent
            update_every (int): how often to train and update the network


        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.lr = lr

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        self.prio_replay = prio_replay
        self.double_ql = double_ql
             
        # Replay memory
        if self.prio_replay:
            self.memory = PrioReplayBuffer(action_size, self.buffer_size, self.batch_size, seed, p_alpha = p_alpha)
        else:
            self.memory = PrioReplayBuffer(action_size, self.buffer_size, self.batch_size, seed, p_alpha = 0.)

        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        double q learning pytorch
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, p = experiences

        # Loss function:
        # L = E[(y-Qlocal(s,a)**2] with y = r + gamma*max_a'[Qtarget(s',a')]       
        Qlocal = self.qnetwork_local(states).gather(1, actions)
        self.qnetwork_target.eval()
        Qtarget = self.qnetwork_target(next_states)

        if not self.double_ql:
            target = rewards + (1.-dones) * gamma * Qtarget.detach().max(1)[0].unsqueeze(1)
        # Double q learning
        else:
            max_action = self.qnetwork_local(next_states).detach().max(1)[1]
            max_action += torch.range(0,(max_action.size()[0]-1)*self.action_size,self.action_size,dtype=torch.long).to(device)
            target = rewards + (1.-dones) * gamma * Qtarget.detach().take(max_action).unsqueeze(1)
        
        # Update priorities in case of prio replay
        if self.prio_replay:
            delta = (Qlocal.detach() - target.detach()).squeeze()
            self.memory.update_prios(delta.cpu().numpy())
        
        loss = F.mse_loss(Qlocal,target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

   
class PrioReplayBuffer():
    """Fixed-size Prioritized replay buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, p_alpha = 1.):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            p_alpha (float): exponent which shapes the probabilty distribution. p_alpha = 1. corresponds to sampling
                             according to priorities, p_alpha = 0. corresponds to sampling from a uniform distribution,
                             i.e. no priorties are applied
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.prios = np.zeros(buffer_size)
        self.p_eps = 1./(buffer_size*10.)
        self.last_sample = None
        self.p_alpha = p_alpha
        
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        if len(self.memory)==1:
            p_max = 1.
        else:
            p_max = self.prios.max()
        self.prios[len(self.memory)-1] = p_max
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # Transforms priorities to a probability distribution
        p =  self.prios[:len(self.memory)]**self.p_alpha
        p /= p.sum()
        p += 1./(len(self.memory)*100.) #Adds a small constant such that experiences with zero td error can be sampled, too
        p /= p.sum()
        
        self.last_sample = np.random.choice(np.arange(len(self.memory)), self.batch_size, replace=False, p =p)
        experiences = [self.memory[l] for l in self.last_sample]
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, p)

    def update_prios(self, delta):
        """Update priorities with TD error"""
        self.prios[self.last_sample] = np.abs(delta)#+self.p_eps

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
class QNetwork(nn.Module):
    """Build a network that maps state -> action values."""

    def __init__(self, state_size, action_size, seed, hidden_layers = [128,128], p_dropout = 0.):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers ([int]): Number of nodes in each hidden layer
            p_dropout (float): probability for dropout layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.p_dropout = p_dropout              
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, state):
        """Defines forward pass through network"""
        x = state
        for each in self.hidden_layers:
            x = F.relu(each(x))
            if self.p_dropout>0.:
                x = self.dropout(x)
        x = self.output(x)
        return x
