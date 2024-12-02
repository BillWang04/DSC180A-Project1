import torch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from src.models.gcn import GCN
from src.models.gat import GAT
import pandas as pd

def evaluate_model_on_dataset(model, loader):
    """Evaluate model performance on a dataset."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    return correct / total

def compare_datasets():
    """Compare model performance across different datasets."""
    # Load datasets
    cora = Planetoid(root='/tmp/Cora', name='Cora')
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
    
    # Test GCN
    gcn_models = {
        'Cora': GCN(hidden_channels=64, num_layers=2),
        'IMDB': GCN(hidden_channels=64, num_layers=2),
        'ENZYMES': GCN(hidden_channels=64, num_layers=2)
    }
    
    # Test GAT
    gat_models = {
        'Cora': GAT(hidden_channels=64, num_layers=2),
        'IMDB': GAT(hidden_channels=64, num_layers=2),
        'ENZYMES': GAT(hidden_channels=64, num_layers=2)
    }
    
    # Training and evaluation
    for model_name, models in [('GCN', gcn_models), ('GAT', gat_models)]:
        for dataset_name, model in models.items():
            if dataset_name == 'Cora':
                data = cora[0]
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                
                # Train on Cora
                for epoch in range(200):
                    model.train()
                    optimizer.zero_grad()
                    out = model(data.x, data.edge_index)
                    loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
                    loss.backward()
                    optimizer.step()
                
                # Evaluate
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    pred = out.argmax(dim=1)
                    acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
            
            elif dataset_name == 'IMDB':
                acc = evaluate_model_on_dataset(model, imdb_loader)
            else:  # ENZYMES
                acc = evaluate_model_on_dataset(model, enzymes_loader)
            
            results['Model'].append(model_name)
            results['Dataset'].append(dataset_name)
            results['Accuracy'].append(acc)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/dataset_comparison.csv', index=False)
    return results_df

if __name__ == "__main__":
    results = compare_datasets()
    print("\nDataset Comparison Results:")
    print(results)
