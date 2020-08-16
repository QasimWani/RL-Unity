# import numpy as np
# from utils import OUNoise, ReplayBuffer
# from actor import Actor
# from critic import Critic
# import random

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# # Parameters for continuous control are taken from DDPG paper.
# # Continuous Control with Deep Reinforcement Learning: https://arxiv.org/pdf/1509.02971.pdf
# LR_CRITIC = 1e-4 #critic learning rate
# LR_ACTOR = 1e-4 #actor learning rate
# GAMMA = 0.99 #discount factor
# WEIGHT_DECAY = 0 #L2 weight decay 
# TAU = 1e-3 #soft target update
# BUFFER_SIZE = int(1e6) #Size of buffer to train from a single step
# MINI_BATCH = 128 #Max length of memory.

# N_LEARN_UPDATES = 10     # number of learning updates
# N_TIME_STEPS = 20       # every n time step do update

# #Enable cuda if available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class Agent():
#     """Main DDPG agent that extracts experiences and learns from them"""
#     def __init__(self, state_size=33, action_size=4, random_seed=0):
#         """
#         Initializes Agent object.
#         @Param:
#         1. state_size: dimension of each state.
#         2. action_size: number of actions.
#         """
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(random_seed)

#         #Actor network
#         self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device) #local model
#         self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device) #target model, TD-target
#         self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR) #initialize optimizer using Adam as regularizer for Actor network.

#         #Critic network
#         self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device) #local model
#         self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device) #target model, TD-target
#         self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY) #initialize optimizer using Adam as regularizer for Critic network.

#         #Noise proccess
#         self.noise = OUNoise(action_size, random_seed) #define Ornstein-Uhlenbeck process

#         #Replay memory
#         self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, MINI_BATCH, random_seed) #define experience replay buffer object

#     def step(self, time_step, state, action, reward, next_state, done):
#         """
#         Saves an experience in the replay memory to learn from using random sampling.
#         @Param:
#         1. state: current state, S.
#         2. action: action taken based on current state.
#         3. reward: immediate reward from state, action.
#         4. next_state: next state, S', from action, a.
#         5. done: (bool) has the episode terminated?
#         Exracted version for trajectory used in calculating the value for an action, a."""

#         self.memory.add(state, action, reward, next_state, done) #append to memory buffer

#         # only learn every n_time_steps
#         if time_step % N_TIME_STEPS != 0:
#             return

#         #check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
#         if(len(self.memory) > MINI_BATCH):
#             for _ in range(N_LEARN_UPDATES):
#                 experience = self.memory.sample()
#                 self.learn(experience)

#     def reset(self):
#         """Resets the noise process to mean"""
#         self.noise.reset()

#     def act(self, state, add_noise=True):
#         """
#         Returns a deterministic action given current state.
#         @Param:
#         1. state: current state, S.
#         2. add_noise: (bool) add bias to agent, default = True (training mode)
#         """
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device) #typecast to torch.Tensor
#         self.actor_local.eval() #set in evaluation mode
#         with torch.no_grad(): #reset gradients
#             action = self.actor_local(state).cpu().data.numpy() #deterministic action based on Actor's forward pass.
#         self.actor_local.train() #set training mode

#         #If training mode, i.e. add_noise = True, add noise to the model to learn a more accurate policy for current state.
#         if(add_noise):
#             action += self.noise.sample()
#         return np.clip(action, -1, 1)
    
#     def learn(self, experiences, gamma=GAMMA):
#         """
#         Learn from a set of experiences picked up from a random sampling of even frequency (not prioritized)
#         of experiences when buffer_size = MINI_BATCH.
#         Updates policy and value parameters accordingly
#         @Param:
#         1. experiences: (Tuple[torch.Tensor]) set of experiences, trajectory, tau. tuple of (s, a, r, s', done)
#         2. gamma: immediate reward hyper-parameter, 0.99 by default.
#         """
#         #Extrapolate experience into (state, action, reward, next_state, done) tuples
#         states, actions, rewards, next_states, dones = experiences

#         #Update Critic network
#         actions_next = self.actor_target(next_states) # Get predicted next-state actions and Q values from target models
#         Q_targets_next = self.critic_target(next_states, actions_next)
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) #  r + γ * Q-values(a,s)

#         # Compute critic loss using MSE
#         Q_expected = self.critic_local(states, actions)
#         critic_loss = F.mse_loss(Q_expected, Q_targets)

#         # Minimize the loss
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1) #clip gradients
#         self.critic_optimizer.step()

#         #Update Actor Network

#         # Compute actor loss
#         actions_pred = self.actor_local(states) #gets mu(s)
#         actor_loss = -self.critic_local(states, actions_pred).mean() #gets V(s,a)
#         # Minimize the loss
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # ----------------------- update target networks ----------------------- #
#         self.soft_update(self.critic_local, self.critic_target, TAU)
#         self.soft_update(self.actor_local, self.actor_target, TAU)                     

#     def soft_update(self, local_model, target_model, tau):
#         """Soft update model parameters. Copies model τ every experience.
#         θ_target = τ*θ_local + (1 - τ)*θ_target
#         Params
#         ======
#             local_model: PyTorch model (weights will be copied from)
#             target_model: PyTorch model (weights will be copied to)
#             tau (float): interpolation parameter 
#         """
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
import numpy as np
import random
import copy

from actor import Actor
from critic import Critic
from utils import OUNoise, ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

N_LEARN_UPDATES = 10     # number of learning updates
N_TIME_STEPS = 20       # every n time step do update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    memory = None
    actor_local = None
    actor_target = None
    actor_optimizer = None

    critic_local = None
    critic_target = None
    critic_optimizer = None

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # initialize Class level Actor Network
        # if Agent.actor_local is None:
        #     Agent.actor_local = Actor(state_size, action_size, random_seed).to(device)
        # if Agent.actor_target is None:
        #     Agent.actor_target = Actor(state_size, action_size, random_seed).to(device)
        # if Agent.actor_optimizer is None:
        #     Agent.actor_optimizer = optim.Adam(Agent.actor_local.parameters(), lr=LR_ACTOR)
        # self.actor_local = Agent.actor_local
        # self.actor_target = Agent.actor_target
        # self.actor_optimizer = Agent.actor_optimizer

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Initilise Class levell Critic Network
        # if Agent.critic_local is None:
        #     Agent.critic_local = Critic(state_size, action_size, random_seed).to(device)
        # if Agent.critic_target is None:
        #     Agent.critic_target = Critic(state_size, action_size, random_seed).to(device)
        # if Agent.critic_optimizer is None:
        #     Agent.critic_optimizer = optim.Adam(Agent.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # self.critic_local = Agent.critic_local
        # self.critic_target = Agent.critic_target
        # self.critic_optimizer = Agent.critic_optimizer

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory - only intitialise once per class
        if Agent.memory is None:
            print("Initialising ReplayBuffer")
            Agent.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, time_step, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        Agent.memory.add(state, action, reward, next_state, done)

        # only learn every n_time_steps
        if time_step % N_TIME_STEPS != 0:
            return

        # Learn, if enough samples are available in memory
        if len(Agent.memory) > BATCH_SIZE:
            for i in range(N_LEARN_UPDATES):
                experiences = Agent.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).unsqueeze(0).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)