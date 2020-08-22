# Common utils for DDPG implementation.
#> 1. ReplayBuffer - Experience Replay class based on DDPG paper.
#> 2. OUNoise - Ornstein-Uhlenbeck process for Noise calculation.
import numpy as np
import copy
import random
import random
from collections import deque, namedtuple

import torch

## ----------------------- ReplayBuffer ----------------------

#Enable cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    """
    Implementation of a fixed size replay buffer as used in DQN algorithms.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initializes the buffer.
        @Param:
        1. action_size: env.action_space.shape[0]
        2. buffer_size: Maximum length of the buffer for extrapolating all experiences into trajectories. default - 1e6 (Source: DeepMind)
        3. batch_size: size of mini-batch to train on. default = 64.
        """
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.replay_memory = deque(maxlen=buffer_size) #Experience replay memory object
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state_1", "state_2", "action_1", "action_2", "reward_1", "reward_2", "next_state_1", "next_state_2", "done_1", "done_2"]) #standard S,A,R,S',done for both agents (therefore, times 2)
        
    def add(self, self_, other):
        """
        Adds an experience to existing memory
        self_ : (state, action, reward, next_state, done)
        other : (state, action, reward, next_state, done)
        """
        trajectory = self.experience(*self_, *other)
        self.replay_memory.append(trajectory)
    
    def sample(self):
        """Randomly picks minibatches within the replay_buffer of size mini_batch"""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        #Self
        states_1 = torch.from_numpy(np.vstack([e.state_1 for e in experiences if e is not None])).float().to(device)
        actions_1 = torch.from_numpy(np.vstack([e.action_1 for e in experiences if e is not None])).float().to(device)
        rewards_1 = torch.from_numpy(np.vstack([e.reward_1 for e in experiences if e is not None])).float().to(device)
        next_states_1 = torch.from_numpy(np.vstack([e.next_state_1 for e in experiences if e is not None])).float().to(device)
        dones_1 = torch.from_numpy(np.vstack([e.done_1 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        #Other agent
        states_2 = torch.from_numpy(np.vstack([e.state_2 for e in experiences if e is not None])).float().to(device)
        actions_2 = torch.from_numpy(np.vstack([e.action_2 for e in experiences if e is not None])).float().to(device)
        rewards_2 = torch.from_numpy(np.vstack([e.reward_2 for e in experiences if e is not None])).float().to(device)
        next_states_2 = torch.from_numpy(np.vstack([e.next_state_2 for e in experiences if e is not None])).float().to(device)
        dones_2 = torch.from_numpy(np.vstack([e.done_2 for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return states_1, states_2, actions_1, actions_2, rewards_1, rewards_2, next_states_1, next_states_2, dones_1, dones_2

    def __len__(self):#override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)


## ----------------------- OUNoise ---------------------------

#This class defines the OUNoise structure taken from Physics used originally for modelling the velocity of a Brownian particle.
#We are using this to setup our noise because it follows the 3 conditions of MDP process and is Gaussian process.
#Read more about Ornstein-Uhlenbeck process at: https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process

#Parameters for theta and sigma taken from Contionous Control for Deep Reinforcement Learning.

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state 