import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
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
        self.output = nn.Linear(hidden_layers_size[-1], action_size)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
#             x = self.dropout(x)
        return F.relu(self.output(x))