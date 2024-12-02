import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, heads=8):
        super(GAT, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        
        # Final convolution with single head
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1))
        
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, batch=None):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index).relu()
        x = self.convs[-1](x, edge_index)
        
        # Apply global pooling for graph-level tasks
        if batch is not None:
            x = global_mean_pool(x, batch)
            
        x = self.lin(x)
        return x