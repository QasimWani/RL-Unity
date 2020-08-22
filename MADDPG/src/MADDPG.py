import numpy as np
import torch

from utils import ReplayBuffer
from agent import Agent
from model import Actor, Critic

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256       # minibatch size
N_TIME_STEPS = 20       # every n time step do update

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    
    def __init__(self, state_size=24, action_size=2, random_seed=0):
        """
        Initializes 2 Agents
        @Param:
        1. state_size: observational space for a single agent.
        2. action_size: action space for a single agent.
        """
        #Collect sizes
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = 2

        self.agent1 = Agent(state_size, action_size, random_seed)
        self.agent2 = Agent(state_size, action_size, random_seed)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, states, rand = False):
        """Agents act with actor_local"""

        #Exploitation
        if(rand == False):
            action1 = self.agent1.act(states[0])
            action2 = self.agent2.act(states[1])
            return [action1, action2]

        #Exploration
        actions = np.random.randn(self.num_agents, self.action_size) 
        actions = np.clip(actions, -1, 1)
        return actions

    def step(self, time_step, state, action, reward, next_state, done):
        """
        Saves an experience in the replay memory to learn from using random sampling.
        @Param:
        1. state: current state, S.
        2. action: action taken based on current state.
        3. reward: immediate reward from state, action.
        4. next_state: next state, S', from action, a.
        5. done: (bool) has the episode terminated?
        Exracted version for trajectory used in calculating the value for an action, a."""

        self.memory.add(state[0], state[1], action[0], action[1], reward[0], reward[1], next_state[0], next_state[1], done[0], done[1]) #append to memory buffer

        # only learn every n_time_steps
        # if time_step % N_TIME_STEPS != 0:
        #     return

        #check if enough samples in buffer. if so, learn from experiences, otherwise, keep collecting samples.
        if(time_step % N_TIME_STEPS == 0 and len(self.memory) > BATCH_SIZE):
            experience = self.memory.sample()
            self.learn(experience)
    
    def learn(self, experiences):
        """Train mini-batch from experiences"""
        states1, states2, actions1, actions2, rewards1, rewards2, next_states1, next_states2, dones1, dones2 = experiences
        
        # next actions (for CRITIC network)
        actions_next1 = self.agent1.actor_target(next_states1)
        actions_next2 = self.agent2.actor_target(next_states2)
        
        # action predictions (for ACTOR network)
        actions_pred1 = self.agent1.actor_local(states1)
        actions_pred2 = self.agent2.actor_local(states2)
        
        #Gather agent 1 properties
        self_ = {"state": states1, "action": actions1, "reward": rewards1, "next_state": next_states1, "done": dones1, "action_next": actions_next1, "predicted_action": actions_pred1}
        #Gather agent 2 properties
        other = {"state": states2, "action": actions2, "reward": rewards2, "next_state": next_states2, "done": dones2, "action_next": actions_next2, "predicted_action": actions_pred2}

        self.agent1.learn(self_, other)
        self.agent2.learn(other, self_)