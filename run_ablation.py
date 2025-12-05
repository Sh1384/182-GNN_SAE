#!/usr/bin/env python3
"""
SAE Feature Ablation Analysis (Boxplot Distribution)

Compares the distribution of GNN errors (MSE) across test graphs in two scenarios:
1. Unablated: GNN inference using full SAE reconstruction.
2. Ablated: GNN inference using SAE reconstruction with specific features zeroed out.

The "Error" is measured relative to the Original GNN's output (Original vs. Reconstructed).

Usage:
    python run_ablation.py --latent_dim 512 --k 16 --feature z496
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle
import networkx as nx

# Import your models
from sparse_autoencoder import SparseAutoencoder
# Ensure GCNModel is importable
try:
    from gnn_train import GCNModel
except ImportError:
    # Placeholder if file missing in local context
    pass

# Setup Directories
ABLATION_DIR = Path("ablations")
ABLATION_DIR.mkdir(exist_ok=True)
(ABLATION_DIR / "results").mkdir(exist_ok=True)
(ABLATION_DIR / "plots").mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Model Loading
# -----------------------------------------------------------------------------

def load_sae_model(latent_dim, k):
    """Load trained SAE model."""
    checkpoint_path = f"checkpoints/sae_latent{latent_dim}_k{k}.pt"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = SparseAutoencoder(input_dim=64, latent_dim=latent_dim, k=k)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def load_gnn_model():
    """Load trained GNN model."""
    try:
        from gnn_train import GCNModel
        gnn_checkpoint_path = "checkpoints/gnn_model.pt"
        if not Path(gnn_checkpoint_path).exists():
            return None

        gnn = GCNModel(input_dim=2, hidden_dim1=248, hidden_dim2=64, output_dim=1, dropout=0.5)
        state_dict = torch.load(gnn_checkpoint_path, weights_only=True)
        gnn.load_state_dict(state_dict)
        gnn.eval()
        return gnn
    except Exception as e:
        print(f"Warning: Could not load GNN model: {e}")
        return None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_feature_indices(feature_spec, latent_dim):
    features = []
    for feat in feature_spec.split(','):
        if feat.strip().startswith('z'):
            try:
                idx = int(feat.strip()[1:]) - 1
                if 0 <= idx < latent_dim: features.append(idx)
            except: pass
    return features

def load_graph_data(graph_id):
    path = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    if not path.exists(): return None
    with open(path, 'rb') as f: G = pickle.load(f)
    W = nx.to_numpy_array(G, weight='weight')
    edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
    edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)
    return edge_index, edge_weight

def evaluate_gnn_output(gnn_model, layer2_activations, edge_index, edge_weight):
    if gnn_model is None: return None
    with torch.no_grad():
        h3 = gnn_model.conv3(layer2_activations, edge_index, edge_weight=edge_weight)
        pred = h3.squeeze(-1)
    return pred

# -----------------------------------------------------------------------------
# Main Analysis
# -----------------------------------------------------------------------------

def run_ablation_experiment(latent_dim, k, ablate_indices, experiment_name):
    print(f"Running Ablation: {experiment_name}")
    
    sae_model = load_sae_model(latent_dim, k)
    gnn_model = load_gnn_model()
    
    # Use TEST set
    with open('outputs/test_graph_ids.json', 'r') as f:
        graph_ids = json.load(f)['graph_ids']

    results = []

    print(f"Processing {len(graph_ids)} graphs...")
    for graph_id in tqdm(graph_ids):
        # 1. Load Original Activations
        act_file = Path(f"outputs/activations/layer2/test/graph_{graph_id}.pt")
        if not act_file.exists(): continue
        original_acts = torch.load(act_file, weights_only=True)

        # 2. Get SAE Reconstructions
        with torch.no_grad():
            # A. Full Reconstruction (Unablated)
            latents_full = sae_model.encode(original_acts)
            reconstructed_full = sae_model.decoder(latents_full)
            
            # B. Ablated Reconstruction
            latents_ablated = latents_full.clone()
            latents_ablated[:, ablate_indices] = 0.0 # Zero out feature
            reconstructed_ablated = sae_model.decoder(latents_ablated)

        # 3. Run GNN Inference
        graph_data = load_graph_data(graph_id)
        if gnn_model and graph_data:
            edge_index, edge_weight = graph_data
            
            # Run GNN on Original (Ground Truth for this test), Full, and Ablated
            out_orig = evaluate_gnn_output(gnn_model, original_acts, edge_index, edge_weight)
            out_full = evaluate_gnn_output(gnn_model, reconstructed_full, edge_index, edge_weight)
            out_ablate = evaluate_gnn_output(gnn_model, reconstructed_ablated, edge_index, edge_weight)

            if out_orig is not None:
                # Calculate MSE Loss for this specific graph
                loss_unablated = torch.mean((out_orig - out_full) ** 2).item()
                loss_ablated = torch.mean((out_orig - out_ablate) ** 2).item()

                results.append({
                    'graph_id': graph_id,
                    'Loss (Unablated)': loss_unablated,
                    'Loss (Ablated)': loss_ablated,
                    'Difference': loss_ablated - loss_unablated
                })

    return pd.DataFrame(results)

def plot_boxplots(df, experiment_name):
    """Plot side-by-side boxplots of the losses."""
    if df.empty: return

    # Melt dataframe for seaborn boxplot compatibility
    df_melted = df.melt(
        id_vars=['graph_id'], 
        value_vars=['Loss (Unablated)', 'Loss (Ablated)'],
        var_name='Condition', 
        value_name='GNN MSE Loss'
    )

    plt.figure(figsize=(8, 6))
    
    # Create Boxplot
    sns.boxplot(data=df_melted, x='Condition', y='GNN MSE Loss', 
                palette=['lightgray', '#e74c3c'], width=0.5, showfliers=False)
    
    # Overlay swarmplot to see density (optional, remove if too slow for big data)
    # sns.swarmplot(data=df_melted, x='Condition', y='GNN MSE Loss', color=".25", size=2, alpha=0.5)

    plt.title(f'Impact of Feature Ablation on GNN Loss\n({experiment_name})', fontsize=14, fontweight='bold')
    plt.ylabel('GNN MSE (vs Original Output)', fontsize=12)
    plt.grid(axis='y', alpha=0.3)

    # Calculate and print stats
    mean_unablated = df['Loss (Unablated)'].mean()
    mean_ablated = df['Loss (Ablated)'].mean()
    
    # Add text annotation
    plt.figtext(0.5, 0.01, 
                f"Mean Loss Increase: {mean_ablated - mean_unablated:.2e}\n"
                f"(Higher 'Ablated' loss = Feature is Important)", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

    save_path = ABLATION_DIR / "plots" / f"{experiment_name}_boxplot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--feature', type=str, required=True, help='e.g. z496')
    args = parser.parse_args()

    ablate_indices = get_feature_indices(args.feature, args.latent_dim)
    experiment_name = f"ablate_{args.feature}"

    df = run_ablation_experiment(args.latent_dim, args.k, ablate_indices, experiment_name)
    plot_boxplots(df, experiment_name)

if __name__ == "__main__":
    main()