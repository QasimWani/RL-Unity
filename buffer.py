import torch
import torch.nn.functional as F
import torch.optim as optim
from dqnetwork import DQNetwork
import random

class Agent():
    """Defines the agent class for DQN using Double Q-learning and Prioritized Experience Replay architecture"""
    def __init__(self, state_size=37, action_size=4, gamma=0.99, lr=0.001):
        """
        Initializes the model.
        ----
        @param:
        1. state_size: size of input # of states.
        2. action_size: size of # of actions.
        3. gamma: discounted return rate.
        4. lr: learning rate for the model.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.input_features = [state_size, 128, 64, 32]
        self.output_features = [128, 64, 32, action_size]
        
        #Q-network : defines the 2 DQN (using doubling Q-learning architecture via fixed Q target)
        self.qnetwork_local = DQNetwork(input_features, output_features)
        self.qnetwork_target = DQNetwork(input_features, output_features)
        
        #define the optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
