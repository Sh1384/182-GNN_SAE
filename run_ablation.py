#!/usr/bin/env python3
"""
SAE Feature Ablation Analysis (3-Way Comparison)

Compares the distribution of GNN errors (MSE) across test graphs in three scenarios:
1. Original: GNN inference using original layer2 activations (no SAE).
2. Full SAE: GNN inference using full SAE reconstruction.
3. Ablated: GNN inference using SAE reconstruction with specific features zeroed out.

The "Error" is measured as MSE between GNN predictions and GROUND TRUTH expression values
on the masked nodes.

Feature:
- Lines are colored by the dominant motif in the graph.
- Line thickness is proportional to the deviation in MSE (thicker = bigger change).
- Legend included at the top.

Usage:
    python run_ablation.py --latent_dim 512 --k 16 --feature z496
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from tqdm import tqdm
import torch
import pickle
import networkx as nx
from scipy import stats

# Import your models
from sparse_autoencoder import SparseAutoencoder
# Ensure GCNModel is importable
try:
    from gnn_train import GCNModel
except ImportError:
    pass

# Setup Directories
ABLATION_DIR = Path("ablations")
ABLATION_DIR.mkdir(exist_ok=True)
(ABLATION_DIR / "results").mkdir(exist_ok=True)
(ABLATION_DIR / "plots").mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Model Loading & Helpers
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

def get_feature_indices(feature_spec, latent_dim):
    features = []
    for feat in feature_spec.split(','):
        if feat.strip().startswith('z'):
            try:
                idx = int(feat.strip()[1:]) - 1
                if 0 <= idx < latent_dim: features.append(idx)
            except: pass
    return features

def load_graph_motif_metadata(graph_id):
    """Load motif metadata for a specific graph."""
    metadata_path = Path(f"virtual_graphs/data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
    if not metadata_path.exists():
        return {}

    df = pd.read_csv(metadata_path, index_col=0)
    # Count nodes in each motif
    motif_counts = df.sum(axis=0).to_dict()
    return motif_counts

def get_dominant_motif(graph_id):
    """Determine the majority motif for a graph."""
    counts = load_graph_motif_metadata(graph_id)
    
    # Map raw column names to display names
    name_map = {
        'feedforward_loop': 'Feedforward Loop',
        'feedback_loop': 'Feedback Loop',
        'single_input_module': 'Single Input Module',
        'cascade': 'Cascade'
    }
    
    # Filter for known motifs and non-zero counts
    valid_counts = {name_map.get(k, k): v for k, v in counts.items() if v > 0 and k in name_map}
    
    if not valid_counts:
        return "Other"
    
    # Return motif with max count
    return max(valid_counts, key=valid_counts.get)

def simulate_expression(W, graph_id, steps=50, gamma=0.3, noise_std=0.01):
    local_seed = 42 + graph_id
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
    path = Path(f"virtual_graphs/data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    if not path.exists(): return None
    with open(path, 'rb') as f:
        G = pickle.load(f)
    W = nx.to_numpy_array(G, weight='weight')
    edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
    edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)
    y_true = torch.tensor(simulate_expression(W, graph_id), dtype=torch.float32)
    rng = np.random.default_rng(42 + graph_id)
    mask = torch.tensor(rng.random(len(G.nodes())) < 0.3, dtype=torch.bool)
    return edge_index, edge_weight, y_true, mask

def evaluate_gnn_output(gnn_model, layer2_activations, edge_index, edge_weight):
    if gnn_model is None: return None
    with torch.no_grad():
        h3 = gnn_model.conv3(layer2_activations, edge_index, edge_weight=edge_weight)
        pred = h3.squeeze(-1)
    return pred

# -----------------------------------------------------------------------------
# Main Analysis Logic
# -----------------------------------------------------------------------------

def run_ablation_experiment(latent_dim, k, ablate_indices, experiment_name, motif_type_filter=None):
    print(f"Running Ablation: {experiment_name}")
    sae_model = load_sae_model(latent_dim, k)
    gnn_model = load_gnn_model()

    with open('outputs/test_graph_ids.json', 'r') as f:
        graph_ids = json.load(f)['graph_ids']

    results = []
    print(f"Processing {len(graph_ids)} graphs...")
    
    for graph_id in tqdm(graph_ids):
        # 1. Determine Motif
        motif_label = get_dominant_motif(graph_id)
        
        # Apply optional filter
        if motif_type_filter and motif_type_filter.lower() != 'all':
            # Simple check if filter string is part of label
            if motif_type_filter.lower() not in motif_label.lower().replace(" ", "_"):
                continue

        # 2. Load Data
        act_file = Path(f"outputs/activations/layer2/test/graph_{graph_id}.pt")
        if not act_file.exists(): continue
        original_acts = torch.load(act_file, weights_only=True)

        # 3. SAE Reconstructions
        with torch.no_grad():
            latents_full = sae_model.encode(original_acts)
            reconstructed_full = sae_model.decoder(latents_full)
            
            latents_ablated = latents_full.clone()
            latents_ablated[:, ablate_indices] = 0.0
            reconstructed_ablated = sae_model.decoder(latents_ablated)

        # 4. GNN Inference
        graph_data = load_graph_data(graph_id)
        if gnn_model and graph_data:
            edge_index, edge_weight, y_true, mask = graph_data
            
            out_original = evaluate_gnn_output(gnn_model, original_acts, edge_index, edge_weight)
            out_full_sae = evaluate_gnn_output(gnn_model, reconstructed_full, edge_index, edge_weight)
            out_ablated = evaluate_gnn_output(gnn_model, reconstructed_ablated, edge_index, edge_weight)

            if out_original is not None:
                loss_original = torch.mean(((out_original - y_true)[mask]) ** 2).item()
                loss_full_sae = torch.mean(((out_full_sae - y_true)[mask]) ** 2).item()
                loss_ablated = torch.mean(((out_ablated - y_true)[mask]) ** 2).item()

                results.append({
                    'graph_id': graph_id,
                    'Motif': motif_label,
                    'Loss (Original)': loss_original,
                    'Loss (Full SAE)': loss_full_sae,
                    'Loss (Ablated)': loss_ablated,
                    'SAE Degradation': loss_full_sae - loss_original,
                    'Ablation Impact': loss_ablated - loss_full_sae
                })

    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------

def plot_boxplots(df, experiment_name):
    """
    Plot 3-way paired plot colored by motif with top legend.
    FIXED: Uses high zorder, thick black borders, and robust data handling to ensure visibility.
    """
    if df.empty:
        print("Dataframe is empty. Skipping plot.")
        return

    # Define Color Palette (Colorblind friendly)
    motif_palette = {
        'Feedforward Loop': '#377eb8',      # Blue
        'Feedback Loop': '#ff7f00',         # Orange
        'Single Input Module': '#4daf4a',   # Green
        'Cascade': '#e41a1c',               # Red
        'Other': '#999999'                  # Grey
    }

    loss_cols = ['Loss (Original)', 'Loss (Full SAE)', 'Loss (Ablated)']
    
    # 1. Validation & Cleaning
    for col in loss_cols:
        if col not in df.columns:
            print(f"Error: Column '{col}' not found in dataframe.")
            return

    # Filter outliers
    df_clean = df.copy()
    threshold = 3.0
    for col in loss_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            # Filter out points outside 3 IQR range
            df_clean = df_clean[~((df_clean[col] < (Q1 - threshold * IQR)) | (df_clean[col] > (Q3 + threshold * IQR)))]

    # Drop rows with NaNs in the loss columns to ensure lines plot correctly
    df_clean = df_clean.dropna(subset=loss_cols)
    print(f"Plotting {len(df_clean)} graphs after cleaning...")

    fig, ax = plt.subplots(figsize=(12, 8))

    positions = [0, 1, 2]
    x_labels = ['Original', 'Full SAE', 'Ablated']

    # 2. Draw Connected Lines (Bottom Layer)
    # zorder=2 ensures they are behind the points
    segments = []
    colors = []
    linewidths = []
    
    for _, row in df_clean.iterrows():
        motif = row.get('Motif', 'Other')
        c = motif_palette.get(motif, '#999999')
        
        y_orig = row['Loss (Original)']
        y_full = row['Loss (Full SAE)']
        y_abl = row['Loss (Ablated)']
        
        # Calculate differences for line thickness
        diff1 = abs(y_full - y_orig)
        diff2 = abs(y_abl - y_full)
        
        segments.append([(0, y_orig), (1, y_full)])
        colors.append(c)
        linewidths.append(diff1)
        
        segments.append([(1, y_full), (2, y_abl)])
        colors.append(c)
        linewidths.append(diff2)

    # Normalize linewidths: Ensure minimum thickness is visible (1.0)
    lw_array = np.array(linewidths)
    if len(lw_array) > 0:
        max_diff = np.percentile(lw_array, 95)
        if max_diff < 1e-9: max_diff = 1.0
        # Scale: Min 1.0, Max 3.5
        normalized_lws = 1.0 + 2.5 * np.clip(lw_array / max_diff, 0, 1)
    else:
        normalized_lws = np.ones(len(segments))

    # alpha=0.4 gives good visibility for the paired lines
    lc = LineCollection(segments, colors=colors, linewidths=normalized_lws, alpha=0.4, zorder=2)
    ax.add_collection(lc)

    # 3. Draw Scatter Points (Middle Layer)
    # zorder=3 sits on top of lines
    for i, col in enumerate(loss_cols):
        point_colors = [motif_palette.get(m, '#999999') for m in df_clean['Motif']]
        ax.scatter([positions[i]] * len(df_clean), df_clean[col], 
                   c=point_colors, alpha=0.5, s=25, zorder=3, edgecolors='white', linewidth=0.3)

    # 4. Draw Boxplots (Top Layer)
    # zorder=10 forces this to the very front, guaranteeing visibility.
    # facecolor=(0,0,0,0) or 'none' ensures the box is transparent.
    # We use .dropna() on the series passed to boxplot for robustness.
    plot_data = [df_clean[col].dropna() for col in loss_cols]
    
    bp = ax.boxplot(plot_data,
                     positions=positions,
                     widths=0.4,
                     patch_artist=True,
                     # Increased linewidth to 2.5 for clarity, guaranteed transparent fill
                     boxprops=dict(facecolor=(0,0,0,0), edgecolor='black', linewidth=2.5), 
                     whiskerprops=dict(color='black', linewidth=2),
                     capprops=dict(color='black', linewidth=2),
                     # Bold red median line
                     medianprops=dict(color='red', linewidth=3), 
                     showfliers=False,
                     zorder=10) # <-- CRITICAL: High zorder for visibility

    # 5. Legend & Aesthetics
    legend_handles = []
    unique_motifs = df_clean['Motif'].unique() if 'Motif' in df_clean.columns else ['Other']
    for motif, color in motif_palette.items():
        if motif in unique_motifs:
            patch = mpatches.Patch(color=color, label=motif)
            legend_handles.append(patch)

    ax.legend(handles=legend_handles, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 1.02), 
              ncol=min(len(legend_handles), 5), 
              frameon=False,
              fontsize=11)

    ax.set_xticks(positions)
    ax.set_xticklabels(x_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('GNN MSE Loss', fontsize=12)
    ax.set_title(f'Feature Ablation Impact by Graph Motif\n({experiment_name})', fontsize=14, pad=40)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    
    save_path = ABLATION_DIR / "plots" / f"{experiment_name}_colored_boxplot.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {save_path}")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--feature', type=str, required=True, help='e.g. z496 or z496,z200')
    parser.add_argument('--motif_type', type=str, default='all', help='Optional filter')
    parser.add_argument('--experiment_name', type=str, default=None, help='Optional experiment name override')
    args = parser.parse_args()

    ablate_indices = get_feature_indices(args.feature, args.latent_dim)

    # Name experiment
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        feat_str = "multi" if "," in args.feature else args.feature
        experiment_name = f"ablate_{feat_str}"

    print(f"Ablating indices: {ablate_indices}")

    # Run Analysis
    df = run_ablation_experiment(args.latent_dim, args.k, ablate_indices, experiment_name, args.motif_type)

    # Save results to CSV
    results_file = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
    df.to_csv(results_file, index=False)
    print(f"\nSaved results to: {results_file}")

    # Calculate Stats
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Overall Impact
    mean_imp = df['Ablation Impact'].mean()
    p_val = stats.wilcoxon(df['Loss (Full SAE)'], df['Loss (Ablated)'])[1]
    
    print(f"Mean Ablation Impact: {mean_imp:.2e} (p={p_val:.2e})")
    
    # Breakdown by Motif
    print("\nImpact by Motif Type:")
    motif_stats = df.groupby('Motif')['Ablation Impact'].agg(['count', 'mean', 'std'])
    print(motif_stats)

    # Plot
    plot_boxplots(df, experiment_name)

if __name__ == "__main__":
    main()