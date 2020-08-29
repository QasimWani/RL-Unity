from agent import Agent
from utils import ReplayBuffer

import torch

#Define constants
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor

class MADDPG():
    def __init__(self, num_agents=2, state_size=24, action_size=2, random_seed=2):
        """
        Initializes multiple agents using DDPG algorithm.
        @Params:
        1. num_agents: number of agents provided by Unity brain
        2. state_size: observational space per agent.
        3. action_size: number of actions per agent.
        4. random_seed: seed value for reproducibility.
        """
        self.num_agents = num_agents
        #Create n agents, where n = num_agents
        self.agents = [Agent(state_size, action_size, random_seed) for _ in range(self.num_agents)]

        #Create shared experience replay memory
        self.memory = ReplayBuffer(action_size, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, seed=random_seed)
    
    def act(self, states, add_noise=True):
        """Perform action for multiple agents. Uses single agent act(), but now MARL"""
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise) #get action from a single agent
            actions.append(action)
        return actions
    
    def reset(self):
        """Reset the noise level of multiple agents"""
        for agent in self.agents:
            agent.reset()
            
    def step(self, states, actions, rewards, next_states, dones):
        """
        Saves an experience in the replay memory to learn from using random sampling.
        @Param:
        1. states: current state, S.
        2. actions: action taken based on current state.
        3. rewards: immediate reward from state, action.
        4. next_states: next state, S', from action, a.
        5. dones: (bool) has the episode terminated?
        Exracted version for trajectory used in calculating the value for an action, a.
        """
        # Save trajectories to Replay buffer
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        #check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
        if(len(self.memory) > BATCH_SIZE):
            for _ in range(self.num_agents):
                experience = self.memory.sample()
                self.learn(experience)
    
    def learn(self, experiences, gamma=GAMMA):
        """Learn from an agents experiences. performs batch learning for multiple agents simultaneously"""
        for agent in self.agents:
            agent.learn(experiences, gamma)
            
    def saveCheckPoints(self, isDone):
        """Save the checkpoint weights of MARL params every 100 or so episodes"""
        if(isDone == False):
            for i, agent in enumerate(self.agents):
                torch.save(agent.actor_local.state_dict(),  f"../models/checkpoint/actor_agent_{i}.pth")
                torch.save(agent.critic_local.state_dict(), f"../models/checkpoint/critic_agent_{i}.pth")
        else:
            for i, agent in enumerate(self.agents):
                torch.save(agent.actor_local.state_dict(),  f"../models/final/actor_agent_{i}.pth")
                torch.save(agent.critic_local.state_dict(), f"../models/final/critic_agent_{i}.pth")
            
    def loadCheckPoints(self, isFinal=False):
        """Loads the checkpoint weight of MARL params"""
        if(isFinal):
            for i, agent in enumerate(self.agents):
                agent.actor_local.load_state_dict(torch.load(f"../models/final/actor_agent_{i}.pth"))
                agent.critic_local.load_state_dict(torch.load(f"../models/final/critic_agent_{i}.pth"))
        else:
            for i, agent in enumerate(self.agents):
                agent.actor_local.load_state_dict(torch.load(f"../models/checkpoint/actor_agent_{i}.pth"))
                agent.critic_local.load_state_dict(torch.load(f"../models/checkpoint/critic_agent_{i}.pth"))