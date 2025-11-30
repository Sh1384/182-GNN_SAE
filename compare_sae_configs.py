#!/usr/bin/env python3
"""
SAE Hyperparameter Comparison Script

Runs motif analysis across different SAE configurations (latent_dim, k combinations)
and summarizes key metrics to identify optimal hyperparameters.

Usage:
    python compare_sae_configs.py

Output:
    - CSV file with comparison metrics
    - Summary table printed to console
    - Recommended configuration based on composite score
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pointbiserialr
from statsmodels.stats.multitest import multipletests
import torch
from sparse_autoencoder import SparseAutoencoder
import warnings
warnings.filterwarnings('ignore')

# Configuration
INPUT_DIM = 64
N_PERMUTATIONS = 1000
SIGNIFICANCE_LEVEL = 0.05

# Hyperparameter grid to test
CONFIGS = [
    # (latent_dim, k, description)
    (128, 16, "Low capacity, moderate sparsity"),
    (128, 8, "Low capacity, high sparsity"),
    (256, 32, "Medium capacity, low sparsity"),
    (256, 16, "Medium capacity, moderate sparsity"),
    (256, 8, "Medium capacity, high sparsity"),
    (512, 32, "High capacity, low sparsity"),
    (512, 16, "High capacity, moderate sparsity"),
    (512, 8, "High capacity, high sparsity"),
    (1024, 32, "Very high capacity, moderate sparsity"),
    (1024, 16, "Very high capacity, high sparsity"),
]

def load_data_and_model(latent_dim, k):
    """Load SAE model and prepare data."""
    # Load model
    checkpoint_path = f"checkpoints/sae_latent{latent_dim}_k{k}.pt"
    if not Path(checkpoint_path).exists():
        return None, None, None

    model = SparseAutoencoder(input_dim=INPUT_DIM, latent_dim=latent_dim, k=k)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test graph IDs
    with open('outputs/test_graph_ids.json', 'r') as f:
        test_graph_ids = json.load(f)['graph_ids']

    # Extract latent representations
    activation_dir = Path("outputs/activations/layer2/test")
    metadata_dir = Path("virtual_graphs/data/all_graphs/graph_motif_metadata")

    all_latents = []
    all_motifs = []

    for graph_id in test_graph_ids:
        act_file = activation_dir / f"graph_{graph_id}.pt"
        metadata_file = metadata_dir / f"graph_{graph_id}_metadata.csv"

        if not act_file.exists() or not metadata_file.exists():
            continue

        activations = torch.load(act_file, weights_only=True)
        with torch.no_grad():
            latents = model.encode(activations)

        latents_np = latents.cpu().numpy()
        df_meta = pd.read_csv(metadata_file, index_col=0)

        if len(df_meta) != latents_np.shape[0]:
            continue

        for node_idx in range(latents_np.shape[0]):
            latent_row = [graph_id, node_idx] + latents_np[node_idx].tolist()
            all_latents.append(latent_row)

            motif_row = df_meta.iloc[node_idx].to_dict()
            motif_row['graph_id'] = graph_id
            motif_row['node_idx'] = node_idx
            all_motifs.append(motif_row)

    # Create DataFrames
    latent_cols = ['graph_id', 'node_idx'] + [f'z{i+1}' for i in range(latent_dim)]
    df_latents = pd.DataFrame(all_latents, columns=latent_cols)
    df_motifs = pd.DataFrame(all_motifs)
    df = pd.merge(df_latents, df_motifs, on=['graph_id', 'node_idx'])

    # Standardize motif column names
    rename_map = {
        'feedforward_loop': 'in_feedforward_loop',
        'feedback_loop': 'in_feedback_loop',
        'single_input_module': 'in_single_input_module',
        'cascade': 'in_cascade',
    }
    for k_old, v in rename_map.items():
        if k_old in df.columns:
            df = df.rename(columns={k_old: v})

    return model, df, latent_dim

def compute_correlations(df, latent_dim):
    """Compute point-biserial correlations."""
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    correlations = []
    for motif in motif_types:
        if motif not in df.columns:
            continue
        for z_col in latent_features:
            if df[z_col].std() == 0:  # Skip constant features
                continue
            corr, pval = pointbiserialr(df[motif], df[z_col])
            correlations.append({
                'feature': z_col,
                'motif': motif,
                'rpb': corr,
                'pval': pval,
                'rpb_abs': abs(corr),
            })

    return pd.DataFrame(correlations)

def permutation_test(df, df_corr, latent_dim, n_permutations=1000):
    """Run permutation testing and compute empirical p-values."""
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    # Store null distributions
    null_distributions = {motif: {f: [] for f in latent_features} for motif in motif_types}

    for perm_idx in range(n_permutations):
        for motif in motif_types:
            if motif not in df.columns:
                continue
            shuffled_labels = df[motif].sample(frac=1, random_state=42+perm_idx).reset_index(drop=True)

            for z_col in latent_features:
                if df[z_col].std() == 0:
                    continue
                corr_perm, _ = pointbiserialr(shuffled_labels, df[z_col])
                null_distributions[motif][z_col].append(corr_perm)

    # Calculate empirical p-values
    df_corr['p_empirical'] = 1.0

    for idx, row in df_corr.iterrows():
        feature = row['feature']
        motif = row['motif']
        obs_rpb_abs = abs(row['rpb'])

        null_dist = null_distributions[motif][feature]
        if len(null_dist) == 0:
            continue

        p_empirical = (np.abs(null_dist) >= obs_rpb_abs).sum() / n_permutations
        df_corr.loc[idx, 'p_empirical'] = p_empirical

    # FDR correction
    reject, pvals_fdr, _, _ = multipletests(df_corr['p_empirical'],
                                            alpha=SIGNIFICANCE_LEVEL,
                                            method='fdr_bh')
    df_corr['p_fdr'] = pvals_fdr
    df_corr['significant_fdr'] = reject

    return df_corr

def compute_precision_recall(df, feature, motif, percentile=95):
    """Compute precision and recall for a feature-motif pair."""
    threshold = np.percentile(df[feature], percentile)
    activated = df[feature] > threshold
    present = df[motif] == 1
    tp = (activated & present).sum()

    precision = tp / activated.sum() if activated.sum() > 0 else 0
    recall = tp / present.sum() if present.sum() > 0 else 0

    return precision, recall

def analyze_configuration(latent_dim, k):
    """Run full analysis for one configuration."""
    print(f"\nAnalyzing: latent_dim={latent_dim}, k={k} ({100*k/latent_dim:.2f}% active)")

    # Load model and data
    model, df, latent_dim = load_data_and_model(latent_dim, k)
    if model is None or df is None:
        print(f"  ⚠ Checkpoint not found, skipping...")
        return None

    # Compute correlations
    df_corr = compute_correlations(df, latent_dim)
    if len(df_corr) == 0:
        print(f"  ⚠ No valid correlations, skipping...")
        return None

    # Permutation testing
    print(f"  Running {N_PERMUTATIONS} permutations...")
    df_corr = permutation_test(df, df_corr, latent_dim, N_PERMUTATIONS)

    # Compute precision/recall for top features
    latent_features = [f'z{i+1}' for i in range(latent_dim)]
    motif_types = ['in_feedforward_loop', 'in_feedback_loop',
                   'in_single_input_module', 'in_cascade']

    precision_recall_results = []
    for motif in motif_types:
        if motif not in df.columns:
            continue
        motif_corrs = df_corr[df_corr['motif'] == motif].nlargest(10, 'rpb_abs')
        for _, row in motif_corrs.iterrows():
            feature = row['feature']
            precision, recall = compute_precision_recall(df, feature, motif)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            precision_recall_results.append({
                'feature': feature,
                'motif': motif,
                'rpb': row['rpb'],
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

    df_pr = pd.DataFrame(precision_recall_results)

    # Calculate summary metrics
    n_significant = df_corr['significant_fdr'].sum()
    n_features_tested = df_corr['feature'].nunique()
    activation_counts = {f'z{i+1}': (df[f'z{i+1}'] > 0).sum() for i in range(latent_dim)}
    n_active_features = sum(1 for count in activation_counts.values() if count > 0)
    dead_feature_rate = 1 - (n_active_features / latent_dim)

    # Top metrics
    max_rpb = df_corr['rpb_abs'].max() if len(df_corr) > 0 else 0
    max_rpb_feature = df_corr.loc[df_corr['rpb_abs'].idxmax()] if len(df_corr) > 0 else None

    sig_corrs = df_corr[df_corr['significant_fdr']]
    max_sig_rpb = sig_corrs['rpb_abs'].max() if len(sig_corrs) > 0 else 0

    best_f1 = df_pr['f1_score'].max() if len(df_pr) > 0 else 0
    best_f1_row = df_pr.loc[df_pr['f1_score'].idxmax()] if len(df_pr) > 0 else None

    best_precision = df_pr['precision'].max() if len(df_pr) > 0 else 0
    best_recall = df_pr['recall'].max() if len(df_pr) > 0 else 0

    # Composite quality score
    significance_rate = n_significant / len(df_corr) if len(df_corr) > 0 else 0
    capacity_utilization = 1 - dead_feature_rate

    composite_score = (
        0.35 * min(significance_rate * 10, 1.0) +  # Significance (capped at 10%)
        0.25 * min(max_sig_rpb / 0.5, 1.0) +        # Effect size (capped at 0.5)
        0.25 * best_f1 +                             # Predictive power
        0.15 * min(capacity_utilization, 1.0)        # Capacity utilization
    )

    results = {
        'latent_dim': latent_dim,
        'k': k,
        'sparsity_pct': 100 * k / latent_dim,
        'n_features_tested': n_features_tested,
        'n_active_features': n_active_features,
        'dead_feature_rate': dead_feature_rate,
        'n_significant': n_significant,
        'significance_rate': significance_rate,
        'max_rpb_abs': max_rpb,
        'max_rpb_feature': max_rpb_feature['feature'] if max_rpb_feature is not None else 'N/A',
        'max_rpb_motif': max_rpb_feature['motif'] if max_rpb_feature is not None else 'N/A',
        'max_sig_rpb_abs': max_sig_rpb,
        'best_f1': best_f1,
        'best_f1_feature': best_f1_row['feature'] if best_f1_row is not None else 'N/A',
        'best_f1_motif': best_f1_row['motif'] if best_f1_row is not None else 'N/A',
        'best_precision': best_precision,
        'best_recall': best_recall,
        'composite_score': composite_score,
    }

    print(f"  ✓ Significant pairs: {n_significant}/{len(df_corr)}")
    print(f"  ✓ Max |rpb|: {max_rpb:.3f}")
    print(f"  ✓ Best F1: {best_f1:.3f}")
    print(f"  ✓ Composite score: {composite_score:.3f}")

    return results

def main():
    """Run analysis for all configurations and generate summary."""
    print("="*70)
    print("SAE HYPERPARAMETER COMPARISON")
    print("="*70)
    print(f"\nTesting {len(CONFIGS)} configurations...")
    print(f"Permutations per config: {N_PERMUTATIONS}")
    print(f"Significance level: FDR < {SIGNIFICANCE_LEVEL}")

    results = []

    for latent_dim, k, description in tqdm(CONFIGS, desc="Configurations"):
        config_result = analyze_configuration(latent_dim, k)
        if config_result is not None:
            config_result['description'] = description
            results.append(config_result)

    if len(results) == 0:
        print("\n⚠ No configurations completed successfully!")
        return

    # Create summary DataFrame
    df_results = pd.DataFrame(results)

    # Sort by composite score
    df_results = df_results.sort_values('composite_score', ascending=False)

    # Save to CSV
    output_file = 'outputs/sae_config_comparison.csv'
    df_results.to_csv(output_file, index=False)
    print(f"\n✓ Saved detailed results to {output_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: Top 5 Configurations by Composite Score")
    print("="*70)

    display_cols = [
        'latent_dim', 'k', 'sparsity_pct', 'n_significant', 'significance_rate',
        'max_sig_rpb_abs', 'best_f1', 'dead_feature_rate', 'composite_score'
    ]

    top5 = df_results.head(5)[display_cols].copy()
    top5['sparsity_pct'] = top5['sparsity_pct'].round(2)
    top5['significance_rate'] = top5['significance_rate'].round(4)
    top5['max_sig_rpb_abs'] = top5['max_sig_rpb_abs'].round(3)
    top5['best_f1'] = top5['best_f1'].round(3)
    top5['dead_feature_rate'] = top5['dead_feature_rate'].round(3)
    top5['composite_score'] = top5['composite_score'].round(3)

    print(top5.to_string(index=False))

    # Recommendations
    best = df_results.iloc[0]
    print("\n" + "="*70)
    print("RECOMMENDED CONFIGURATION")
    print("="*70)
    print(f"\n  latent_dim = {int(best['latent_dim'])}")
    print(f"  k = {int(best['k'])}")
    print(f"  Sparsity: {best['sparsity_pct']:.2f}% ({best['description']})")
    print(f"\n  Key Metrics:")
    print(f"    • Significant features: {int(best['n_significant'])} ({100*best['significance_rate']:.1f}%)")
    print(f"    • Max correlation: |rpb| = {best['max_sig_rpb_abs']:.3f}")
    print(f"    • Best F1 score: {best['best_f1']:.3f}")
    print(f"    • Active features: {int(best['n_active_features'])}/{int(best['latent_dim'])} ({100*(1-best['dead_feature_rate']):.1f}%)")
    print(f"    • Composite score: {best['composite_score']:.3f}")

    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
  Composite Score Components:
    • 35% - Significance rate (% of features with FDR < 0.05)
    • 25% - Max significant correlation (effect size)
    • 25% - Best F1 score (predictive performance)
    • 15% - Capacity utilization (1 - dead feature rate)

  Good Configuration Indicators:
    ✓ Composite score > 0.5
    ✓ Significance rate > 2-5%
    ✓ Max |rpb| > 0.3
    ✓ Best F1 > 0.3
    ✓ Dead feature rate < 0.5

  To use the recommended configuration, update your notebook:
    LATENT_DIM = {best_latent}
    K = {best_k}
    """.format(best_latent=int(best['latent_dim']), best_k=int(best['k'])))

if __name__ == "__main__":
    main()
