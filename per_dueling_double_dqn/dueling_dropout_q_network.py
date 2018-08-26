import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDropoutQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        hidden_layers_size = [state_size,128,64,32]
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.extend([nn.Linear(hidden_layers_size[i], hidden_layers_size[i + 1]) for i in range(len(hidden_layers_size) - 1)])
        self.advantage1 = nn.Linear(hidden_layers_size[-1],16)
        self.advantage2 = nn.Linear(16, action_size)
        
        self.value1 = nn.Linear(hidden_layers_size[-1],16)
        self.value2 = nn.Linear(16,1)
        self.dropout = nn.Dropout(0.2)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
        x_advantage = F.relu(self.advantage1(x))
        x_advantage = F.relu(self.advantage2(x_advantage))
        x_value = self.value1(x)
        x_value = self.value2(x_value)
        # Q = V + A - mean(A)
        # size of Q will be (batch, action_size)
        # size of V will be (batch, 1)
        # size of A will be (batch, action_size)
        # size of mean(A) will be (batch, 1)
        # since this model allows us to have the Q of all actions of a given state
        # therefore, we need to apply the above formula for each of them
        mean_advantage = x_advantage.mean(1).unsqueeze(1).expand_as(x_advantage)
        x_value = x_value.expand_as(x_advantage)
        return x_value + x_advantage - mean_advantage



