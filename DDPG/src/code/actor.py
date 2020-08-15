import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Estimates the policy deterministically using tanh activation for continuous action space"""
    def __init__(self, state_size=33, action_size=4, fc1=400, fc2=300):
        """
        @Param:
        1. state_size: number of observations, i.e. brain.vector_action_space_size
        2. action_size: number of actions, i.e. env_info.vector_observations.shape[1]
        3. fc1: number of hidden units in the first fully connected layer. Default = 400.
        4. fc2: number of hidden units in the second fully connected layer, default = 300.
        """
        super(Actor, self).__init__()
        #Layer 1
        self.fc1 = nn.Linear(state_size, fc1)
        self.bn1 = nn.BatchNorm1d(fc1)
        #Layer 2
        self.fc2 = nn.Linear(fc1, fc2) 
        self.bn2 = nn.BatchNorm1d(fc2)
        #Output layer
        self.mu = nn.Linear(fc2, action_size) # µ(s|θ) {Deterministic policy}
        
        #Initialize Weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters by setting a noise from distribution following from its respective hidden unit size.
        Format for (-fx,fx) followed from the original paper.
        """
        
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)
        
    def forward(self, state):
        """
        Performs a single forward pass to map (state,action) to policy, pi.
        @Param:
        1. state: current observations, shape: (env.observation_space.shape[0],)
        2. action: immediate action to evaluate against, shape: (env.action_space.shape[0],)
        @Return:
        - µ(s|θ)
        """
        x = state
        #Layer #1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        #Layer #2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        #Output
        x = self.mu(x)
        mu = torch.tanh(x)
        return mu