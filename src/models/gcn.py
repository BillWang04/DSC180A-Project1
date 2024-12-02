import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GCN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        
        # Apply global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)  # [num_nodes, hidden_channels] -> [num_graphs, hidden_channels]
        
        x = self.lin(x)
        return x