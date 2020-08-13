import numpy as np
import torch
import torch.optim as optim
from dqnetwork import DQNetwork
import random
from buffer import ReplayBuffer
import torch.nn.functional as F


#Define constants
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


class Agent():
    """Defines the agent class for DQN using Double Q-learning and Prioritized Experience Replay architecture"""
    def __init__(self, state_size=37, action_size=4, gamma=0.99, lr=0.001, update_every=5):
        """
        Initializes the model.
        ----
        @param:
        1. state_size: size of input # of states.
        2. action_size: size of # of actions.
        3. gamma: discounted return rate.
        4. lr: learning rate for the model.
        5. update_every: update target_model every X time-steps.
        """
        self.state_size = state_size
        self.action_size = action_size
        
        self.gamma = gamma #define dicsounted return
        
        #Q-network : defines the 2 DQN (using doubling Q-learning architecture via fixed Q target)
        self.qnetwork_local = DQNetwork()
        self.qnetwork_target = DQNetwork()
        
        #define the optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        
        #replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.update_every = update_every
        self.target_update_counter = 0
        
    def step(self, transition):
        """Performs forward pass of the tranisition
        @param:
        1. transition : (tuple) state, action, reward, next_state, done
        """
        # Save experience in replay memory
        self.memory.add(transition)
        self.target_update_counter = (self.target_update_counter + 1) % self.update_every #cyclic update
        
        #Update target network to local network
        if(self.target_update_counter == 0 and self.memory.isSampling() == True):
            experiences = self.memory.sample()
            self.train(experiences)
    
    def get_action(self, state, eps=0.0):
        """
        Determines the action to perform based on epsilon-greedy method
        @param:
        1. state - list of current observations to determine an action for
        2. eps - value for epsilon, stochastic measure.
        @return:
        - action = action chosen by either equiprobably π or using Q-table
        """
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval() #set in evaluation mode
        with torch.no_grad():
            action_val = self.qnetwork_local(state)
        
        self.qnetwork_local.train()#set in training mode
        
        #Epsilon-greedy selection
        if(random.random() > eps):#exploit
            return np.argmax(action_val.cpu().data.numpy())
        
        return random.choice(np.arange(self.action_size))#explore
    
    def train(self, experiences):
        """
        Train the model.
        @param:
        1. experiences: (Tuple[torch.Variable]) (s,a,r,s',done)
        """
        states, actions, rewards, next_states, done = experiences
        
        #Implement SGD using Adam as regularizer
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - done))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        #set loss as mse.
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #Update target network using soft update
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self, local_model, target_model):
        """
        Update target network to local network using a soft update param, τ.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        ------
        @param:
        1. local_model: (DQNetwork) local network model (weights will be copied from)
        2. target_model: (DQNetwork) target network model (weights will be copied into)
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
