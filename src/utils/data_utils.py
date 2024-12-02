import torch
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree
import numpy as np

def load_and_preprocess_data(dataset, normalize=True):
    """
    Load and preprocess a PyG dataset.
    
    Args:
        dataset: PyG dataset object
        normalize: Whether to normalize features
    
    Returns:
        Processed dataset
    """
    if normalize:
        dataset.transform = NormalizeFeatures()
    
    data = dataset[0]
    
    # Add self-loops if not present
    if 'edge_index' in data:
        loop_edge_index = torch.arange(data.num_nodes)
        loop_edge_index = loop_edge_index.unsqueeze(0).repeat(2, 1)
        data.edge_index = torch.cat([data.edge_index, loop_edge_index], dim=1)
    
    return data

def analyze_graph_statistics(data):
    """
    Analyze and return basic statistics about the graph.
    
    Args:
        data: PyG data object
    
    Returns:
        dict: Graph statistics
    """
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1),
        'num_features': data.num_features,
        'num_classes': data.num_classes if hasattr(data, 'num_classes') else None,
        'avg_degree': degree(data.edge_index[0]).mean().item(),
        'density': (2 * data.edge_index.size(1)) / (data.num_nodes * (data.num_nodes - 1))
    }
    
    if hasattr(data, 'train_mask'):
        stats.update({
            'train_samples': data.train_mask.sum().item(),
            'val_samples': data.val_mask.sum().item() if hasattr(data, 'val_mask') else None,
            'test_samples': data.test_mask.sum().item()
        })
    
    return stats

def create_train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/validation/test split for a dataset.
    
    Args:
        data: PyG data object
        train_ratio: Ratio of training samples
        val_ratio: Ratio of validation samples
    
    Returns:
        data: Updated PyG data object with masks
    """
    num_samples = data.num_nodes
    indices = torch.randperm(num_samples)
    
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    data.train_mask = torch.zeros(num_samples, dtype=torch.bool)
    data.val_mask = torch.zeros(num_samples, dtype=torch.bool)
    data.test_mask = torch.zeros(num_samples, dtype=torch.bool)
    
    data.train_mask[train_indices] = True
    data.val_mask[val_indices] = True
    data.test_mask[test_indices] = True
    
    return data
