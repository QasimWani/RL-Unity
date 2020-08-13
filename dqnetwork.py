import torch
import torch.nn as nn
import torch.nn.functional as F

#set seed
torch.manual_seed(0)


class DQNetwork(nn.Module):
    """
    Defines the feed forward NN used for the DQN Agent.
    inherits nn.modules class.
    """
    def __init__(self, state_size=37, action_size=4, fc1=128, fc2=64, fc3=32):
        """
        Initializes the model.
        ------
        @Param:
        1. input_features: list of input dimensions for the NN.
        2. output_features: list of corresponding output dimensions.
        3. dropout_layers: list of dropout layers; keep_probs value (stochastic) of length < num_layers
        """
        super(DQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, fc1)
        self.dp1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(fc1, fc2)
        self.dp2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, action_size) #output layer
        
    def forward(self, state):
        """
        Build a network that performs feed forward for one pass, by mapping input_space, S -> action, A.
        Returns the corresponding model.
        -------
        @param:
        1. state: (array_like) input state.
        @Return:
        - model: the corresponding model
        """
        x = state
        #Layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dp1(x)#apply first dropout
        #Layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dp2(x)#apply second dropout
        #Layer 3
        x = self.fc3(x)
        x = F.relu(x)
        #Output layer
        x = self.fc4(x)
        return x #dim = self.action_size
        
        
        