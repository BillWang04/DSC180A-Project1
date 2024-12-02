import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import accuracy_score
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..models.gcn import GCN 
from ..models.gat import GAT


def train_and_evaluate_single_graph(model, data, optimizer, criterion, epochs=200):
    """Training for single graph datasets (like Cora)"""
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    train_acc = accuracy_score(data.y[data.train_mask].cpu(), pred[data.train_mask].cpu())
    test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    return train_acc, test_acc

def train_and_evaluate_graph_dataset(model, train_loader, test_loader, optimizer, criterion, epochs=200):
    """Training for multiple graph datasets (like IMDB-BINARY)"""
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    
    test_acc = correct / total
    return test_acc

def run_experiments():
    # Load datasets
    print("Loading datasets...")
    datasets = {
        'Cora': Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
    }

    model_classes = {
        'GCN': GCN,
        'GAT': GAT
    }

    results = []
    
    for model_name, model_class in model_classes.items():
        for dataset_name, dataset in datasets.items():
            print(f"\nRunning {model_name} experiments on {dataset_name}")
            
            # Determine input features and number of classes
            if dataset_name == 'Cora':
                num_features = dataset.num_features
                num_classes = dataset.num_classes
            else:
                num_features = 1  # Use constant feature for graph classification
                num_classes = dataset.num_classes
                
            # Try different numbers of layers
            for num_layers in range(2, 14):
                print(f"Testing with {num_layers} layers")
                
                model = model_class(
                    in_channels=num_features,
                    hidden_channels=64,
                    out_channels=num_classes,
                    num_layers=num_layers
                )
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                criterion = torch.nn.CrossEntropyLoss()
                
                if dataset_name == 'Cora':
                    train_acc, test_acc = train_and_evaluate_single_graph(
                        model, dataset[0], optimizer, criterion
                    )
                    
                    results.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'num_layers': num_layers,
                        'train_acc': train_acc,
                        'test_acc': test_acc
                    })
                    
                    print(f'Layers: {num_layers}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
                
                else:
                    # Create train/test split for graph datasets
                    train_size = int(0.8 * len(dataset))
                    test_size = len(dataset) - train_size
                    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
                    
                    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=32)
                    
                    test_acc = train_and_evaluate_graph_dataset(
                        model, train_loader, test_loader, optimizer, criterion
                    )
                    
                    results.append({
                        'model': model_name,
                        'dataset': dataset_name,
                        'num_layers': num_layers,
                        'test_acc': test_acc
                    })
                    
                    print(f'Layers: {num_layers}, Test Acc: {test_acc:.4f}')

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/figures/layer_analysis_all_models_datasets.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    for dataset_name in datasets.keys():
        dataset_results = results_df[
            (results_df['dataset'] == dataset_name) & 
            (results_df['model'] == 'GCN')
        ]
        plt.plot(dataset_results['num_layers'], 
                dataset_results['test_acc'], 
                marker='o', 
                label=dataset_name)
    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.title('GCN Performance Across Datasets')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    for dataset_name in datasets.keys():
        dataset_results = results_df[
            (results_df['dataset'] == dataset_name) & 
            (results_df['model'] == 'GAT')
        ]
        plt.plot(dataset_results['num_layers'], 
                dataset_results['test_acc'], 
                marker='o', 
                label=dataset_name)
    plt.xlabel('Number of Layers')
    plt.ylabel('Test Accuracy')
    plt.title('GAT Performance Across Datasets')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/layer_model_comparison.png')
    plt.close()

    return results_df

if __name__ == "__main__":
    results = run_experiments()
    print("\nFinal Results:")
    print(results)