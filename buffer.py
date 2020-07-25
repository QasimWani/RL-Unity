import random

class ReplayBuffer():
    """Defines the standard fixed size Experience Replay"""
    def __init__(self, buffer_size, batch_size):
        """
        Initializes a ReplayBuffer object
        ---
        @param:
        1. buffer_size: (int) max. length of the buffer (usually a deque or heap)
        2. batch_size: (int) size of the buffer. usually, 32 or 64.
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        #Experience Replay init
        self.replay_memory = deque(maxlen=self.buffer_size) #initialize experience replay buffer (circular)
        
    def add(self, transition):
        """
        Appends to the underlying replay memory.
        ---
        @param:
        1. transition: (tuple) set of state-action value pair.
            when extracted, (state, action, reward, next_state, done)
        """
        self.replay_memory.append(transition) #store observed state-action tuples in replay memory.
    
    def sample(self):
        """
        Gausian based shuffling for retrieving experiences from the replay_memory.
        """
        experiences = random.sample(self.replay_memory, k=(self.batch_size if self.isSampling() else len(self.replay_memory)))
        return tuple(zip(*experiences)) #unzips into individual states, actions, rewards, next_actions, done(s)
    
    def isSampling(self):
        """Determines if sampling condition has been met, i.e. len(memory) > num_batches"""
        return self.batch_size < len(self.replay_memory)
