import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, heads=4):
        super(GAT, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(dataset.num_node_features, hidden_channels, heads=heads, concat=True))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
        
        self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False))
        
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)
    
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.lin(x)

def train_and_evaluate(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=1)
    train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()
    
    return train_acc.item(), test_acc.item()
