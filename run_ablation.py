#!/usr/bin/env python3
"""
SAE Feature Ablation Analysis

Zero out specific latent features in the SAE, reconstruct layer2 activations, and measure
impact on downstream GNN performance.

Workflow:
1. Load trained SAE model (from checkpoints/sae_latent{dim}_k{k}.pt)
2. Load trained GNN model (from checkpoints/gnn_model.pt)
3. For each test graph:
   - Load original layer2 activations (64-dim)
   - Encode to SAE latent space
   - Zero out specified features (e.g., z69, z200)
   - Reconstruct activations via SAE decoder
   - Run GNN layer3 with both original and ablated activations
   - Compare GNN loss to measure impact

Usage:
    # Ablate a specific feature
    python run_ablation.py --latent_dim 512 --k 32 --feature z69

    # Ablate multiple specific features
    python run_ablation.py --latent_dim 512 --k 32 --feature z69,z200

    # Ablate top N most correlated features (requires motif analysis results)
    python run_ablation.py --latent_dim 512 --k 32 --top_n 10

Output:
    - Ablated activations: ablations/activations/{experiment_name}/graph_{id}.pt
    - Results CSV: ablations/results/{experiment_name}_results.csv
    - Visualizations: ablations/plots/{experiment_name}_summary.png
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
from sparse_autoencoder import SparseAutoencoder

# Create ablations directory structure
ABLATION_DIR = Path("ablations")
ABLATION_DIR.mkdir(exist_ok=True)
(ABLATION_DIR / "activations").mkdir(exist_ok=True)
(ABLATION_DIR / "results").mkdir(exist_ok=True)
(ABLATION_DIR / "plots").mkdir(exist_ok=True)

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
            print(f"Warning: GNN checkpoint not found at {gnn_checkpoint_path}")
            return None

        # Load GCN model (3-layer: 2 -> 128 -> 64 -> 1)
        gnn = GCNModel(
            input_dim=2,
            hidden_dim1=128,
            hidden_dim2=64,
            output_dim=1,
            dropout=0.2
        )

        # Load state dict
        state_dict = torch.load(gnn_checkpoint_path, weights_only=True)
        gnn.load_state_dict(state_dict)
        gnn.eval()

        return gnn
    except Exception as e:
        print(f"Warning: Could not load GNN model: {e}")
        return None

def get_feature_indices(feature_spec, latent_dim):
    """
    Parse feature specification into list of feature indices.

    Args:
        feature_spec: str like "z154" or "z154,z298" or None
        latent_dim: int, total number of latent features

    Returns:
        list of int: feature indices (0-indexed)
    """
    if feature_spec is None:
        return []

    features = []
    for feat in feature_spec.split(','):
        feat = feat.strip()
        if feat.startswith('z'):
            idx = int(feat[1:]) - 1  # Convert z154 -> 153 (0-indexed)
            if 0 <= idx < latent_dim:
                features.append(idx)
            else:
                print(f"Warning: Feature {feat} out of range (0-{latent_dim-1}), skipping")

    return features

def get_top_correlated_features(latent_dim, k, top_n):
    """
    Get top N most correlated features from the motif analysis.

    Args:
        latent_dim: int
        k: int
        top_n: int, number of top features to return

    Returns:
        list of int: feature indices (0-indexed)
    """
    # Load correlation results if they exist
    corr_file = Path("outputs/feature_motif_correlations_configurable.csv")
    if not corr_file.exists():
        print(f"Warning: Correlation file not found at {corr_file}")
        print("Run the motif analysis notebook first to identify top features.")
        return []

    df_corr = pd.read_csv(corr_file, index_col=0)

    # Get features with highest absolute correlation across all motifs
    max_corrs = df_corr.abs().max(axis=1).nlargest(top_n)

    # Convert feature names (z1, z2, ...) to indices (0, 1, ...)
    feature_indices = [int(feat[1:]) - 1 for feat in max_corrs.index]

    print(f"\nTop {top_n} most correlated features:")
    for idx, feat_name in enumerate(max_corrs.index):
        print(f"  {feat_name}: max |rpb| = {max_corrs.iloc[idx]:.3f}")

    return feature_indices

def ablate_and_reconstruct(sae_model, activations, ablate_indices, latent_dim):
    """
    Zero out specified latent features and reconstruct activations.

    Args:
        sae_model: trained SAE model
        activations: torch.Tensor [num_nodes, 64]
        ablate_indices: list of int, which latent features to zero out (0-indexed)
        latent_dim: int

    Returns:
        reconstructed: torch.Tensor [num_nodes, 64], reconstructed activations
        latents_original: torch.Tensor [num_nodes, latent_dim]
        latents_ablated: torch.Tensor [num_nodes, latent_dim]
    """
    with torch.no_grad():
        # Encode to latent space
        latents_original = sae_model.encode(activations)

        # Create ablated version
        latents_ablated = latents_original.clone()
        latents_ablated[:, ablate_indices] = 0.0  # Zero out specified features

        # Decode back to activation space
        reconstructed = sae_model.decoder(latents_ablated)

    return reconstructed, latents_original, latents_ablated

def compute_reconstruction_error(original, reconstructed):
    """Compute MSE between original and reconstructed activations."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    return mse

def load_graph_data(graph_id):
    """Load graph structure for GNN evaluation."""
    import pickle

    # Load graph from pickle file
    graph_file = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    if not graph_file.exists():
        return None

    with open(graph_file, 'rb') as f:
        G = pickle.load(f)

    # Convert to PyG format
    import networkx as nx
    W = nx.to_numpy_array(G, weight='weight')

    # Create edge_index and edge_weights from adjacency matrix
    edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
    edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

    return edge_index, edge_weight

def evaluate_gnn_output(gnn_model, layer2_activations, edge_index, edge_weight):
    """
    Get GNN output using provided layer2 activations.

    This runs only layer3 of the GNN with the provided activations.

    Args:
        gnn_model: trained GCN model
        layer2_activations: torch.Tensor [num_nodes, 64] - layer2 activations
        edge_index: torch.Tensor [2, num_edges]
        edge_weight: torch.Tensor [num_edges]

    Returns:
        torch.Tensor [num_nodes] - GNN predictions
    """
    if gnn_model is None:
        return None

    with torch.no_grad():
        # Run only layer3 of the GNN with provided layer2 activations
        # Layer3: GCNConv(64 -> 1)
        h3 = gnn_model.conv3(layer2_activations, edge_index, edge_weight=edge_weight)
        pred = h3.squeeze(-1)

    return pred

def run_ablation_experiment(latent_dim, k, ablate_indices, experiment_name):
    """
    Run full ablation experiment.

    Args:
        latent_dim: int
        k: int
        ablate_indices: list of int, features to ablate
        experiment_name: str, identifier for this experiment

    Returns:
        DataFrame with results
    """
    print(f"\n{'='*70}")
    print(f"ABLATION EXPERIMENT: {experiment_name}")
    print(f"{'='*70}")
    print(f"SAE config: latent_dim={latent_dim}, k={k}")
    print(f"Ablating features: {[f'z{i+1}' for i in ablate_indices]}")

    # Load models
    print("\nLoading models...")
    sae_model = load_sae_model(latent_dim, k)
    gnn_model = load_gnn_model()

    # Load test graph IDs
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']

    # Create directory for ablated activations
    ablated_act_dir = ABLATION_DIR / "activations" / experiment_name
    ablated_act_dir.mkdir(exist_ok=True, parents=True)

    results = []

    print(f"\nProcessing {len(test_graph_ids)} test graphs...")
    for graph_id in tqdm(test_graph_ids, desc="Ablation progress"):
        # Load original activations
        act_file = Path(f"outputs/activations/layer2/test/graph_{graph_id}.pt")
        if not act_file.exists():
            continue

        original_acts = torch.load(act_file, weights_only=True)

        # Ablate and reconstruct
        reconstructed_acts, latents_orig, latents_ablated = ablate_and_reconstruct(
            sae_model, original_acts, ablate_indices, latent_dim
        )

        # Save ablated activations
        torch.save(reconstructed_acts, ablated_act_dir / f"graph_{graph_id}.pt")

        # Compute reconstruction error
        recon_error = compute_reconstruction_error(original_acts, reconstructed_acts)

        # Compute change in latent activations
        latent_change = torch.mean(torch.abs(latents_orig - latents_ablated)).item()
        n_affected_nodes = (torch.abs(latents_orig - latents_ablated).sum(dim=1) > 0).sum().item()

        # Evaluate GNN output change (if GNN model available)
        graph_data = load_graph_data(graph_id)

        gnn_output_mse = None

        if gnn_model is not None and graph_data is not None:
            edge_index, edge_weight = graph_data

            # Get GNN outputs with original and ablated activations
            output_original = evaluate_gnn_output(
                gnn_model, original_acts, edge_index, edge_weight
            )
            output_ablated = evaluate_gnn_output(
                gnn_model, reconstructed_acts, edge_index, edge_weight
            )

            # Compute MSE between original and ablated outputs
            if output_original is not None and output_ablated is not None:
                gnn_output_mse = torch.mean((output_original - output_ablated) ** 2).item()

        # Store results
        result = {
            'graph_id': graph_id,
            'experiment': experiment_name,
            'n_ablated_features': len(ablate_indices),
            'ablated_features': ','.join([f'z{i+1}' for i in ablate_indices]),
            'reconstruction_mse': recon_error,
            'latent_change_mean': latent_change,
            'n_affected_nodes': n_affected_nodes,
            'total_nodes': original_acts.shape[0],
            'gnn_output_mse': gnn_output_mse,  # MSE between original and ablated GNN outputs
        }

        results.append(result)

    df_results = pd.DataFrame(results)

    # Save results
    results_file = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
    df_results.to_csv(results_file, index=False)
    print(f"\n✓ Saved results to {results_file}")

    return df_results

def plot_ablation_results(df_results, experiment_name):
    """Create visualization of ablation results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Reconstruction error distribution
    ax = axes[0, 0]
    ax.hist(df_results['reconstruction_mse'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel('Reconstruction MSE', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Reconstruction Error After Ablation', fontsize=14, fontweight='bold')
    ax.axvline(df_results['reconstruction_mse'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {df_results["reconstruction_mse"].mean():.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Nodes affected
    ax = axes[0, 1]
    affected_pct = 100 * df_results['n_affected_nodes'] / df_results['total_nodes']
    ax.hist(affected_pct, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('% Nodes Affected', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Percentage of Nodes Affected by Ablation', fontsize=14, fontweight='bold')
    ax.axvline(affected_pct.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean = {affected_pct.mean():.1f}%')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: GNN output change (if available)
    ax = axes[1, 0]
    if df_results['gnn_output_mse'].notna().any():
        ax.hist(df_results['gnn_output_mse'].dropna(), bins=50, alpha=0.7,
                color='green', edgecolor='black')
        ax.set_xlabel('GNN Output MSE (Original vs Ablated)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Impact on GNN Output', fontsize=14, fontweight='bold')
        ax.axvline(df_results['gnn_output_mse'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Mean = {df_results["gnn_output_mse"].mean():.6f}')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'GNN evaluation not available',
                ha='center', va='center', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Plot 4: Scatter - Reconstruction error vs GNN output change
    ax = axes[1, 1]
    if df_results['gnn_output_mse'].notna().any():
        ax.scatter(df_results['reconstruction_mse'], df_results['gnn_output_mse'],
                   alpha=0.6, s=40, color='purple')
        ax.set_xlabel('Reconstruction MSE', fontsize=12)
        ax.set_ylabel('GNN Output MSE', fontsize=12)
        ax.set_title('Reconstruction Error vs GNN Output Change', fontsize=14, fontweight='bold')

        # Add correlation
        corr = df_results[['reconstruction_mse', 'gnn_output_mse']].corr().iloc[0, 1]
        ax.text(0.95, 0.95, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, fontsize=11, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'GNN evaluation not available',
                ha='center', va='center', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = ABLATION_DIR / "plots" / f"{experiment_name}_summary.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_file}")
    plt.close()

def print_summary(df_results, experiment_name):
    """Print summary statistics for ablation experiment."""
    print(f"\n{'='*70}")
    print(f"ABLATION SUMMARY: {experiment_name}")
    print(f"{'='*70}")

    print(f"\nGraphs processed: {len(df_results)}")
    print(f"Features ablated: {df_results['ablated_features'].iloc[0]}")

    print(f"\nReconstruction Error:")
    print(f"  Mean MSE: {df_results['reconstruction_mse'].mean():.6f}")
    print(f"  Std MSE:  {df_results['reconstruction_mse'].std():.6f}")
    print(f"  Median MSE: {df_results['reconstruction_mse'].median():.6f}")

    affected_pct = 100 * df_results['n_affected_nodes'] / df_results['total_nodes']
    print(f"\nNodes Affected:")
    print(f"  Mean: {affected_pct.mean():.2f}%")
    print(f"  Std:  {affected_pct.std():.2f}%")

    if df_results['gnn_output_mse'].notna().any():
        print(f"\nGNN Output Change:")
        print(f"  Mean MSE: {df_results['gnn_output_mse'].mean():.6f}")
        print(f"  Std MSE:  {df_results['gnn_output_mse'].std():.6f}")
        print(f"  Median MSE: {df_results['gnn_output_mse'].median():.6f}")

        # Interpretation
        mean_mse = df_results['gnn_output_mse'].mean()
        if mean_mse > 0.001:
            impact = "SIGNIFICANT CHANGE (ablated feature affects GNN predictions!)"
        elif mean_mse > 0.0001:
            impact = "MODERATE CHANGE (some impact on GNN)"
        else:
            impact = "MINIMAL CHANGE (feature is redundant for GNN)"
        print(f"  Interpretation: {impact}")
    else:
        print(f"\nGNN Output Change: Not evaluated (GNN model not available)")

def main():
    parser = argparse.ArgumentParser(description='SAE Feature Ablation Analysis')
    parser.add_argument('--latent_dim', type=int, required=True,
                        help='SAE latent dimension')
    parser.add_argument('--k', type=int, required=True,
                        help='SAE TopK sparsity parameter')
    parser.add_argument('--feature', type=str, default=None,
                        help='Feature(s) to ablate (e.g., "z154" or "z154,z298")')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Ablate top N most correlated features')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name (auto-generated if not provided)')

    args = parser.parse_args()

    # Determine which features to ablate
    if args.feature is not None:
        ablate_indices = get_feature_indices(args.feature, args.latent_dim)
        if not ablate_indices:
            print("Error: No valid features specified!")
            return
    elif args.top_n is not None:
        ablate_indices = get_top_correlated_features(args.latent_dim, args.k, args.top_n)
        if not ablate_indices:
            print("Error: Could not identify top features!")
            return
    else:
        print("Error: Must specify either --feature or --top_n")
        return

    # Generate experiment name
    if args.experiment_name is None:
        feature_str = '_'.join([f'z{i+1}' for i in ablate_indices])
        experiment_name = f"latent{args.latent_dim}_k{args.k}_ablate_{feature_str}"
    else:
        experiment_name = args.experiment_name

    # Run ablation
    df_results = run_ablation_experiment(
        args.latent_dim,
        args.k,
        ablate_indices,
        experiment_name
    )

    # Create visualizations
    plot_ablation_results(df_results, experiment_name)

    # Print summary
    print_summary(df_results, experiment_name)

    print(f"\n{'='*70}")
    print("ABLATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: ablations/results/{experiment_name}_results.csv")
    print(f"Ablated activations saved to: ablations/activations/{experiment_name}/")
    print(f"Plots saved to: ablations/plots/{experiment_name}_summary.png")

if __name__ == "__main__":
    main()
