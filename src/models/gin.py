import torch
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GIN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        
        # First layer
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn1))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn))
        
        # Final layer
        nn_final = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn_final))
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        # Apply global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        x = self.lin(x)
        return x