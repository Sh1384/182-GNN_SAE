#!/usr/bin/env python3
"""
SAE Feature Ablation Analysis (3-Way Comparison)

Compares the distribution of GNN errors (MSE) across test graphs in three scenarios:
1. Original: GNN inference using original layer2 activations (no SAE).
2. Full SAE: GNN inference using full SAE reconstruction.
3. Ablated: GNN inference using SAE reconstruction with specific features zeroed out.

The "Error" is measured as MSE between GNN predictions and GROUND TRUTH expression values
on the masked nodes (same evaluation as during training).

Usage:
    # Single feature
    python run_ablation.py --latent_dim 512 --k 16 --feature z496

    # Multiple features
    python run_ablation.py --latent_dim 512 --k 16 --feature z496,z200,z123
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
from scipy import stats

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

def simulate_expression(W, graph_id, steps=50, gamma=0.3, noise_std=0.01):
    """
    Simulate gene expression dynamics (same as training).
    Uses consistent seed per graph_id to match training data.
    """
    # Use same seed as training: base_seed(42) + graph_id
    local_seed = 42 + graph_id  # TEST dataset uses seed=44, but we need consistency
    rng = np.random.default_rng(local_seed)

    n_nodes = W.shape[0]
    x = rng.uniform(0, 1, size=n_nodes)

    for _ in range(steps):
        weighted_input = W @ x
        sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
        noise = rng.normal(0, noise_std, size=n_nodes)
        x = (1 - gamma) * x + gamma * sigmoid_input + noise
        x = np.clip(x, 0, 1)

    return x

def load_graph_data(graph_id):
    """Load graph structure and generate ground truth expression."""
    path = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    if not path.exists(): return None

    with open(path, 'rb') as f:
        G = pickle.load(f)

    W = nx.to_numpy_array(G, weight='weight')
    edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
    edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

    # Generate ground truth expression (same as training)
    y_true = simulate_expression(W, graph_id)
    y_true = torch.tensor(y_true, dtype=torch.float32)

    # Generate mask (same as training: 30% masked, consistent per graph)
    mask_prob = 0.3
    rng = np.random.default_rng(42 + graph_id)
    mask = torch.tensor(rng.random(len(G.nodes())) < mask_prob, dtype=torch.bool)

    return edge_index, edge_weight, y_true, mask

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

        # 3. Load Ground Truth and Run GNN Inference
        graph_data = load_graph_data(graph_id)
        if gnn_model and graph_data:
            edge_index, edge_weight, y_true, mask = graph_data

            # 4. Run GNN layer3 with THREE different inputs
            out_original = evaluate_gnn_output(gnn_model, original_acts, edge_index, edge_weight)
            out_full_sae = evaluate_gnn_output(gnn_model, reconstructed_full, edge_index, edge_weight)
            out_ablated = evaluate_gnn_output(gnn_model, reconstructed_ablated, edge_index, edge_weight)

            if out_original is not None and out_full_sae is not None and out_ablated is not None:
                # 5. Calculate all three losses vs ground truth on MASKED nodes only
                loss_original = torch.mean(((out_original - y_true)[mask]) ** 2).item()
                loss_full_sae = torch.mean(((out_full_sae - y_true)[mask]) ** 2).item()
                loss_ablated = torch.mean(((out_ablated - y_true)[mask]) ** 2).item()

                # 6. Store all metrics
                results.append({
                    'graph_id': graph_id,
                    'Loss (Original)': loss_original,
                    'Loss (Full SAE)': loss_full_sae,
                    'Loss (Ablated)': loss_ablated,
                    'SAE Degradation': loss_full_sae - loss_original,
                    'Ablation Impact': loss_ablated - loss_full_sae,
                    'Total Impact': loss_ablated - loss_original
                })

    return pd.DataFrame(results)

def plot_boxplots(df, experiment_name):
    """Plot 3-way paired plot with connected lines showing per-graph trajectories."""
    if df.empty: return

    # Remove extreme outliers using IQR method
    def remove_outliers(df, columns, threshold=3.0):
        """Remove rows where any column has values beyond threshold * IQR"""
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        return df_clean

    # Remove outliers
    loss_cols = ['Loss (Original)', 'Loss (Full SAE)', 'Loss (Ablated)']
    df_clean = remove_outliers(df, loss_cols, threshold=2.5)
    n_removed = len(df) - len(df_clean)
    print(f"Removed {n_removed} outlier graphs ({n_removed/len(df)*100:.1f}%)")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Positions for the three conditions
    positions = [0, 1, 2]
    x_labels = ['Original', 'Full SAE', 'Ablated']

    # Draw boxplots (minimal, just for reference)
    bp = ax.boxplot([df_clean['Loss (Original)'], df_clean['Loss (Full SAE)'], df_clean['Loss (Ablated)']],
                     positions=positions,
                     widths=0.4,
                     patch_artist=True,
                     boxprops=dict(facecolor='lightgray', alpha=0.3),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='gray', linewidth=1),
                     capprops=dict(color='gray', linewidth=1),
                     showfliers=False)

    # Draw connected scatter points for each graph
    for idx, row in df_clean.iterrows():
        values = [row['Loss (Original)'], row['Loss (Full SAE)'], row['Loss (Ablated)']]
        ax.plot(positions, values, 'o-', color='steelblue', alpha=0.15, linewidth=0.5, markersize=3)

    # Add scatter points on top
    for i, col in enumerate(loss_cols):
        ax.scatter([positions[i]] * len(df_clean), df_clean[col],
                   alpha=0.3, s=20, color='steelblue', zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel('GNN MSE (vs Ground Truth)', fontsize=12)
    ax.set_title(f'3-Way Paired Ablation: {experiment_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = ABLATION_DIR / "plots" / f"{experiment_name}_boxplot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")

def plot_impact_decomposition(df, experiment_name):
    """Show SAE degradation vs Ablation impact."""
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Distribution of impacts
    df_impacts = df.melt(
        id_vars=['graph_id'],
        value_vars=['SAE Degradation', 'Ablation Impact'],
        var_name='Impact Type',
        value_name='Loss Increase'
    )

    sns.boxplot(ax=axes[0], data=df_impacts, x='Impact Type', y='Loss Increase')
    axes[0].set_title('Impact Distribution')
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    # Right: Scatter (SAE degradation vs Ablation impact)
    axes[1].scatter(df['SAE Degradation'], df['Ablation Impact'], alpha=0.5)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('SAE Degradation')
    axes[1].set_ylabel('Ablation Impact')
    axes[1].set_title('Per-Graph Impact')

    # Add diagonal line (ablation impact = SAE degradation)
    max_val = max(df['SAE Degradation'].max(), df['Ablation Impact'].max())
    min_val = min(df['SAE Degradation'].min(), df['Ablation Impact'].min())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equal Impact')
    axes[1].legend()

    plt.tight_layout()
    save_path = ABLATION_DIR / "plots" / f"{experiment_name}_decomposition.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Decomposition plot saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--feature', type=str, required=True, help='e.g. z496 or z496,z200,z123')
    args = parser.parse_args()

    ablate_indices = get_feature_indices(args.feature, args.latent_dim)

    # Create experiment name based on number of features
    feature_list = [f.strip() for f in args.feature.split(',') if f.strip()]
    if len(feature_list) == 1:
        experiment_name = f"ablate_{feature_list[0]}"
    else:
        experiment_name = f"ablate_{len(feature_list)}features"

    print(f"Ablating {len(ablate_indices)} feature(s): {ablate_indices}")

    df = run_ablation_experiment(args.latent_dim, args.k, ablate_indices, experiment_name)

    # Save results
    results_path = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"3-WAY ABLATION SUMMARY: {experiment_name}")
    print(f"{'='*70}")
    print(f"Graphs analyzed: {len(df)}")

    mean_orig = df['Loss (Original)'].mean()
    mean_full = df['Loss (Full SAE)'].mean()
    mean_ablated = df['Loss (Ablated)'].mean()
    std_orig = df['Loss (Original)'].std()
    std_full = df['Loss (Full SAE)'].std()
    std_ablated = df['Loss (Ablated)'].std()

    sae_deg = mean_full - mean_orig
    ablation_imp = mean_ablated - mean_full
    total_imp = mean_ablated - mean_orig

    print(f"\nMean Losses (±std):")
    print(f"  Original (No SAE):     {mean_orig:.6f} ± {std_orig:.6f}")
    print(f"  Full SAE:              {mean_full:.6f} ± {std_full:.6f}")
    print(f"  Ablated SAE:           {mean_ablated:.6f} ± {std_ablated:.6f}")

    print(f"\nMean Impact Breakdown:")
    print(f"  SAE Degradation:       {sae_deg:.6f}")
    print(f"  Ablation Impact:       {ablation_imp:.6f}")
    print(f"  Total Impact:          {total_imp:.6f}")

    if abs(sae_deg) > 1e-10:
        rel_importance = ablation_imp / sae_deg
        print(f"  Relative Importance:   {rel_importance:.2%} of SAE degradation")

    # Statistical significance tests (Wilcoxon signed-rank)
    _, pval_sae = stats.wilcoxon(df['Loss (Original)'], df['Loss (Full SAE)'])
    _, pval_ablation = stats.wilcoxon(df['Loss (Full SAE)'], df['Loss (Ablated)'])
    _, pval_total = stats.wilcoxon(df['Loss (Original)'], df['Loss (Ablated)'])

    print(f"\nStatistical Significance (Wilcoxon signed-rank test):")
    print(f"  Original vs Full SAE:  p = {pval_sae:.4e}")
    print(f"  Full SAE vs Ablated:   p = {pval_ablation:.4e}")
    print(f"  Original vs Ablated:   p = {pval_total:.4e}")
    print(f"{'='*70}")

    plot_boxplots(df, experiment_name)
    plot_impact_decomposition(df, experiment_name)

if __name__ == "__main__":
    main()