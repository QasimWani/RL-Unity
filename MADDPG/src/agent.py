import numpy as np
from utils import OUNoise
from model import Actor, Critic
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Parameters for continuous control are taken from DDPG paper.
# Continuous Control with Deep Reinforcement Learning: https://arxiv.org/pdf/1509.02971.pdf

LR_CRITIC = 1e-4 #critic learning rate
LR_ACTOR = 1e-4 #actor learning rate
GAMMA = 0.99 #discount factor
WEIGHT_DECAY = 0 #L2 weight decay 
TAU = 1e-3 #soft target update
BUFFER_SIZE = int(1e6) #Size of buffer to train from a single step
MINI_BATCH = 128 #Max length of memory.

#Enable cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Main DDPG agent that extracts experiences and learns from them"""
    def __init__(self, state_size=24, action_size=2, random_seed=0):
        """
        Initializes Agent object.
        @Param:
        1. state_size: dimension of each state.
        2. action_size: number of actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        #Actor network
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        #Critic network
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        #Noise proccess
        self.noise = OUNoise(action_size, random_seed) #define Ornstein-Uhlenbeck process

    def act(self, state, add_noise=True):
        """
        Returns a deterministic action given current state.
        @Param:
        1. state: current state, S of self_ agent.
        2. add_noise: (bool) add bias to agent, default = True (training mode)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device) #typecast to torch.Tensor
        self.actor_local.eval() #set in evaluation mode
        with torch.no_grad(): #reset gradients
            action = self.actor_local(state).cpu().data.numpy() #deterministic action based on Actor's forward pass.
        self.actor_local.train() #set training mode

        #If training mode, i.e. add_noise = True, add noise to the model to learn a more accurate policy for current state.
        if(add_noise):
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def reset(self):
        """Resets the noise process to mean"""
        self.noise.reset()
    
    def learn(self, self_, other):
        """
        Learn from a set of experiences picked up from a random sampling of even frequency (not prioritized)
        of experiences when buffer_size = MINI_BATCH.
        Updates policy and value parameters accordingly
        @Param:
        1. self_: Agent 1 - dict(state, action, reward, next_state, done, next_action, predicted_action)
        2. other: Agent 2 - dict(state, action, reward, next_state, done, next_action, predicted_action)
        """
        with torch.no_grad():
            Q_targets_next = self.critic_target((self_["state"], other["state"]), (self_["action_next"], other["action_next"]))
        
        #Update Critic network
        Q_targets = self_["reward"] + (GAMMA * Q_targets_next * (1 - self_["done"])) #  r + γ * Q-values(a,s)

        # Compute critic loss using MSE
        Q_expected = self.critic_local((self_["state"], other["state"]), (self_["action"], other["action"]))
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True) #save buffer
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic_local((self_["state"], other["state"]), (self_["predicted_action"], other["predicted_action"])).mean() #gets V(s,a)
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        print(actor_loss)
        actor_loss.backward(retain_graph=True) #save buffer
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters. Copies model τ every experience.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)