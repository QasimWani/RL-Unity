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
    def __init__(self, input_features, output_features, dropout_layers=[0.3, 0.1]):
        """
        Initializes the model.
        ------
        @Param:
        1. input_features: list of input dimensions for the NN.
        2. output_features: list of corresponding output dimensions.
        3. dropout_layers: list of dropout layers; keep_probs value (stochastic) of length < num_layers
        """
        super().__init__()
        self.state_size = input_features[0]#size of observational space
        self.action_size = output_features[-1] #size of action space
        self.FC = nn.ModuleList()#initialize list of FC layers
        self.Dropout = []#intitialize list of dropout layers
        
        #check to see if input_dim = output_dim
        if(len(input_features) != len(output_features)):
            raise ValueError("lengths do not match. input dimension MUST equal output dimensions")
        
        #check to see if dropout dim = L - 1:
        if(len(dropout_layers) >= len(input_features) - 1):
            raise ValueError("dropout layers dimensions do not match appropriate size")
            
        for input_unit, output_unit in zip(input_features, output_features):
            self.FC.append(nn.Linear(input_unit, output_unit, bias=True))#add Linear layers to the network
        
        #set dropout layers
        for prob in dropout_layers:
            self.Dropout.append(nn.Dropout(prob))#append dropout layers with keep_probs to NN.
            
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
        state = torch.from_numpy(state).float().unsqueeze(0)
        for i in range(len(self.FC)):
            if(i == 0):
                X = F.relu(self.FC[0](state))
            else:
                X = F.relu(self.FC[i](X)) if (i < len(self.FC) - 1) else self.FC[i](X)
            if(i < len(self.Dropout)):
                X = self.Dropout[i](X)
        return X
