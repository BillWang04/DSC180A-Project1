import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.gcn import GCN
from src.models.gat import GAT
from torch_geometric.datasets import Planetoid

def analyze_layer_impact(model_class, dataset, max_layers=13, hidden_channels=64):
    """
    Analyze the impact of increasing layers on model performance.
    
    Args:
        model_class: GNN model class (GCN or GAT)
        dataset: PyG dataset
        max_layers: Maximum number of layers to test
        hidden_channels: Number of hidden channels
    """
    layer_counts = np.arange(2, max_layers)
    train_accuracies = []
    test_accuracies = []
    
    for num_layers in layer_counts:
        model = model_class(hidden_channels=hidden_channels, num_layers=num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(dataset.x, dataset.edge_index)
            loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            optimizer.step()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(dataset.x, dataset.edge_index)
            pred = out.argmax(dim=1)
            train_acc = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).sum() / dataset.train_mask.sum()
            test_acc = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum() / dataset.test_mask.sum()
            
        train_accuracies.append(train_acc.item())
        test_accuracies.append(test_acc.item())
        print(f'Layers: {num_layers}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
    
    return layer_counts, train_accuracies, test_accuracies

def plot_layer_analysis(layer_counts, train_accuracies, test_accuracies, model_name):
    """Plot the results of layer analysis."""
    plt.figure(figsize=(10, 6))
    plt.plot(layer_counts, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(layer_counts, test_accuracies, label='Test Accuracy', marker='o')
    plt.xlabel('Number of Layers')
    plt.ylabel('Accuracy')
    plt.title(f'Effect of Increasing Layers on {model_name} Performance')
    plt.legend()
    plt.savefig(f'results/figures/{model_name.lower()}_layer_analysis.png')
    plt.close()

if __name__ == "__main__":
    # Load dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]
    
    # Analyze GCN
    gcn_layers, gcn_train_acc, gcn_test_acc = analyze_layer_impact(GCN, data)
    plot_layer_analysis(gcn_layers, gcn_train_acc, gcn_test_acc, "GCN")
    
    # Analyze GAT
    gat_layers, gat_train_acc, gat_test_acc = analyze_layer_impact(GAT, data)
    plot_layer_analysis(gat_layers, gat_train_acc, gat_test_acc, "GAT")
