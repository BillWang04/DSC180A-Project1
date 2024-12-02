# DSC180A Project Proposal

This repository contains implementations and experiments with different Graph Neural Network (GNN) architectures, focusing on the effects of layer depth and performance across different datasets.

## Contents
- Implementation of GCN (Graph Convolutional Network)
- Implementation of GAT (Graph Attention Network)
- Layer depth analysis
- Experiments on multiple datasets (Cora, IMDB-BINARY, ENZYMES)

## Project Structure
```
.
├── README.md
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── __init__.py
|   |   ├── gin.py
│   │   ├── gcn.py
│   │   └── gat.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── run_experiments.py
│   │   └── dataset_comparison.py
└── results/
    └── figures/
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gnn-experiments.git
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

## Requirements
- Python 3.8+
- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib
- NumPy
- Scikit-learn
- Pandas

## Experiments

### Layer Analysis
We investigate the impact of increasing the number of layers in GNN architectures:
- GCN performance with varying depths
- GAT performance with varying depths
- Analysis of oversquashing and oversmoothing effects

```bash
python -m src.experiments.run_experiments
```


### Dataset Experiments
Performance comparison across different datasets:
- Cora (Citation Network)
- IMDB-BINARY (Movie Collaboration)
- ENZYMES (Protein Structures)

```bash
python -m src.experiments.dataset_comparison
```

## Results

Key findings:
1. Performance degradation observed with increasing layers for both GCN and GAT
2. Dataset-specific performance characteristics
3. Empirical analysis of optimal layer depths


