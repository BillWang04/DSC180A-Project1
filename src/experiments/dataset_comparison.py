import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.gin import GIN

def evaluate_model_on_dataset(model, loader):
    """Evaluate model performance on a dataset."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        with torch.no_grad():
            # Create constant feature for datasets without node features
            if batch.x is None:
                batch.x = torch.ones((batch.num_nodes, 1))
            
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

def compare_models():
    """Compare model performance across different datasets."""
    # Load datasets
    cora = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
    imdb = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
    enzymes = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
    
    # Create dataloaders
    imdb_loader = DataLoader(imdb, batch_size=32, shuffle=True)
    enzymes_loader = DataLoader(enzymes, batch_size=32, shuffle=True)
    
    results = {
        'Model': [],
        'Dataset': [],
        'Accuracy': []
    }
    
    # Initialize models with appropriate input dimensions
    models = {
        'GCN': lambda in_f, out_f: GCN(in_channels=in_f, hidden_channels=64, out_channels=out_f, num_layers=2),
        'GAT': lambda in_f, out_f: GAT(in_channels=in_f, hidden_channels=64, out_channels=out_f, num_layers=2),
        'GIN': lambda in_f, out_f: GIN(in_channels=in_f, hidden_channels=64, out_channels=out_f, num_layers=2)
    }
    
    # Test each model on each dataset
    for model_name, model_fn in models.items():
        print(f"\nTesting {model_name}")
        
        # Test on Cora
        model = model_fn(cora.num_features, cora.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train on Cora
        for epoch in range(200):
            model.train()
            optimizer.zero_grad()
            out = model(cora[0].x, cora[0].edge_index)
            loss = torch.nn.functional.cross_entropy(out[cora[0].train_mask], cora[0].y[cora[0].train_mask])
            loss.backward()
            optimizer.step()
        
        # Evaluate Cora
        model.eval()
        with torch.no_grad():
            out = model(cora[0].x, cora[0].edge_index)
            pred = out.argmax(dim=1)
            acc = (pred[cora[0].test_mask] == cora[0].y[cora[0].test_mask]).sum().item() / cora[0].test_mask.sum().item()
        
        results['Model'].append(model_name)
        results['Dataset'].append('Cora')
        results['Accuracy'].append(acc)
        print(f"Cora Accuracy: {acc:.4f}")
        
        # Test on IMDB
        # Use single feature for IMDB dataset
        model = model_fn(1, imdb.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train on IMDB
        for epoch in range(200):
            model.train()
            for batch in imdb_loader:
                optimizer.zero_grad()
                if batch.x is None:
                    batch.x = torch.ones((batch.num_nodes, 1))
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.functional.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
        
        acc = evaluate_model_on_dataset(model, imdb_loader)
        results['Model'].append(model_name)
        results['Dataset'].append('IMDB')
        results['Accuracy'].append(acc)
        print(f"IMDB Accuracy: {acc:.4f}")
        
        # Test on ENZYMES
        model = model_fn(enzymes.num_features, enzymes.num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Train on ENZYMES
        for epoch in range(200):
            model.train()
            for batch in enzymes_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = torch.nn.functional.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
        
        acc = evaluate_model_on_dataset(model, enzymes_loader)
        results['Model'].append(model_name)
        results['Dataset'].append('ENZYMES')
        results['Accuracy'].append(acc)
        print(f"ENZYMES Accuracy: {acc:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best model for each dataset
    best_models = results_df.loc[results_df.groupby('Dataset')['Accuracy'].idxmax()]
    
    print("\nBest performing models:")
    for _, row in best_models.iterrows():
        print(f"{row['Dataset']}: {row['Model']} (Accuracy: {row['Accuracy']:.4f})")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    datasets = results_df['Dataset'].unique()
    x = np.arange(len(datasets))
    width = 0.25
    
    for i, model in enumerate(models.keys()):
        accuracies = [results_df[(results_df['Model'] == model) & (results_df['Dataset'] == d)]['Accuracy'].values[0] 
                     for d in datasets]
        plt.bar(x + i * width, accuracies, width, label=model)
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width, datasets)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    
    return results_df, best_models

if __name__ == "__main__":
    results, best_models = compare_models()
    
    # Save results
    results.to_csv('results/figures/model_comparison.csv', index=False)
    best_models.to_csv('results/figures/best_models.csv', index=False)
    
    print("\nResults have been saved to CSV files")