#!/usr/bin/env python3
"""
Motif-Specific SAE Feature Ablation Experiments (k=4 variant)

Updated for corrected metadata with actual motif detection.
Uses latent_dim=512, k=4 (instead of k=16).

Expected changes with corrected metadata:
- More balanced feature counts across motifs
- Multi-label encoding means features may correlate with multiple motifs
- Cascade and feedforward loop now have significant features

Usage:
    python run_interpretability_experiments_k4.py --latent_dim 512 --k 4 --n_random_trials 20
"""

import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
import sys

# Directories
ABLATION_DIR = Path("ablations")
RESULTS_DIR = ABLATION_DIR / "interpretability_k4_results"
PLOTS_DIR = ABLATION_DIR / "interpretability_k4_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_correlation_data():
    """Load correlation data and extract significant features per motif."""
    df = pd.read_csv('outputs/latent_correlations.csv', index_col=0)

    # Get significant features for each motif
    sig_df = df[df['significant_fdr'] == True]

    # Group by motif
    motif_features = {}
    motif_map = {
        'in_cascade': 'Cascade',
        'in_feedback_loop': 'Feedback Loop',
        'in_feedforward_loop': 'Feedforward Loop',
        'in_single_input_module': 'Single Input Module'
    }

    for motif_raw, motif_name in motif_map.items():
        features = sig_df[sig_df['motif'] == motif_raw]['feature'].tolist()
        motif_features[motif_name] = sorted(features)

    # Get all non-significant features for random controls
    all_features = df['feature'].unique()
    sig_features = set(sig_df['feature'].unique())
    non_sig_features = sorted([f for f in all_features if f not in sig_features])

    return motif_features, non_sig_features

def save_feature_mapping(motif_features):
    """Save feature mapping to JSON."""
    output_path = Path('outputs/feature_motif_mapping_k4.json')
    with open(output_path, 'w') as f:
        json.dump(motif_features, f, indent=2)
    print(f"Saved feature mapping to: {output_path}")

def run_single_ablation(latent_dim, k, features, experiment_name):
    """Run a single ablation experiment."""
    feature_str = ','.join(features)

    cmd = [
        sys.executable, "run_ablation.py",
        "--latent_dim", str(latent_dim),
        "--k", str(k),
        "--feature", feature_str,
        "--experiment_name", experiment_name
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error in ablation {experiment_name}:")
        print(result.stderr)
        return None

    # Load results
    results_file = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
    if not results_file.exists():
        return None

    df = pd.read_csv(results_file)
    return df

def run_random_control_trials(latent_dim, k, n_features, n_trials, non_sig_features):
    """Run multiple random ablation trials and collect statistics."""
    print(f"\nRunning {n_trials} random control trials (ablating {n_features} features each)...")

    trial_results = []

    for trial in tqdm(range(n_trials), desc=f"Random trials ({n_features} features)"):
        # Sample random non-significant features
        if len(non_sig_features) < n_features:
            print(f"Warning: Only {len(non_sig_features)} non-sig features available, need {n_features}")
            random_features = non_sig_features
        else:
            random_features = sorted(np.random.choice(non_sig_features, n_features, replace=False))

        experiment_name = f"ablate_random_trial_{n_features}feat_n{trial}_k{k}"

        df = run_single_ablation(latent_dim, k, random_features, experiment_name)

        if df is not None:
            # Compute mean impact per motif type for this trial
            for motif in df['Motif'].unique():
                motif_data = df[df['Motif'] == motif]
                trial_results.append({
                    'trial': trial,
                    'n_features': n_features,
                    'motif': motif,
                    'mean_impact': motif_data['Ablation Impact'].mean(),
                    'median_impact': motif_data['Ablation Impact'].median(),
                    'std_impact': motif_data['Ablation Impact'].std()
                })

    return pd.DataFrame(trial_results)

def plot_comparison_with_random_controls(motif_specific_results, random_results_dict, motif_features):
    """Generate comprehensive comparison plots."""

    motif_colors = {
        'Feedforward Loop': '#377eb8',
        'Feedback Loop': '#ff7f00',
        'Single Input Module': '#4daf4a',
        'Cascade': '#e41a1c'
    }

    # Determine which motifs to plot (only those with significant features)
    motifs_to_plot = [m for m, feats in motif_features.items() if len(feats) > 0]
    n_motifs = len(motifs_to_plot)

    if n_motifs == 0:
        print("No motifs with significant features to plot!")
        return

    # Create subplots
    fig, axes = plt.subplots(1, n_motifs, figsize=(6*n_motifs, 6))
    if n_motifs == 1:
        axes = [axes]

    motif_order = ['Feedforward Loop', 'Feedback Loop', 'Single Input Module', 'Cascade']
    x_pos = np.arange(len(motif_order))
    width = 0.35

    for idx, motif_name in enumerate(motifs_to_plot):
        ax = axes[idx]
        n_features = len(motif_features[motif_name])

        # Get specific motif results
        specific = motif_specific_results[motif_specific_results['Feature_Set'] == motif_name]
        random = random_results_dict[n_features]

        # Specific features
        specific_means = [specific[specific['Motif'] == m]['Mean_Impact'].values[0]
                         if len(specific[specific['Motif'] == m]) > 0 else 0
                         for m in motif_order]
        specific_ses = [specific[specific['Motif'] == m]['SE_Impact'].values[0]
                       if len(specific[specific['Motif'] == m]) > 0 else 0
                       for m in motif_order]

        # Random controls
        random_means = [random[random['motif'] == m]['mean_impact'].mean()
                       if len(random[random['motif'] == m]) > 0 else 0
                       for m in motif_order]
        random_ses = [random[random['motif'] == m]['mean_impact'].std() / np.sqrt(len(random[random['motif'] == m]))
                     if len(random[random['motif'] == m]) > 0 else 0
                     for m in motif_order]

        bars1 = ax.bar(x_pos - width/2, specific_means, width,
                      label=f'{motif_name} Features ({n_features})',
                      yerr=specific_ses, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5,
                      color=[motif_colors.get(m, '#999999') for m in motif_order])
        bars2 = ax.bar(x_pos + width/2, random_means, width,
                      label=f'Random Controls ({n_features} feat, 20 trials)',
                      yerr=random_ses, capsize=5, alpha=0.6, edgecolor='black', linewidth=1.5,
                      color='gray')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(motif_order, rotation=45, ha='right')
        ax.set_ylabel('Mean Ablation Impact\n(MSE increase)', fontsize=11)
        ax.set_title(f'{motif_name} Features vs Random Controls\n({n_features} features ablated)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('SAE Interpretability: Motif-Specific Features vs Random Controls (k=4)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'interpretability_vs_random_controls_k4.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'interpretability_vs_random_controls_k4.png'}")
    plt.close()

def compute_statistical_significance(motif_specific_results, random_results):
    """Compute statistical tests comparing motif-specific to random."""
    results = []

    for feature_set in motif_specific_results['Feature_Set'].unique():
        specific_data = motif_specific_results[motif_specific_results['Feature_Set'] == feature_set]

        # Match random controls by number of features
        n_features = specific_data['N_Features'].values[0]
        random_data = random_results[random_results['n_features'] == n_features]

        for motif in specific_data['Motif'].unique():
            specific_impact = specific_data[specific_data['Motif'] == motif]['Mean_Impact'].values[0]
            random_impacts = random_data[random_data['motif'] == motif]['mean_impact'].values

            if len(random_impacts) > 0:
                # Percentile of specific impact in random distribution
                percentile = stats.percentileofscore(random_impacts, specific_impact)

                # Z-score
                random_mean = random_impacts.mean()
                random_std = random_impacts.std()
                z_score = (specific_impact - random_mean) / random_std if random_std > 0 else 0

                # One-sample t-test
                t_stat, p_val = stats.ttest_1samp(random_impacts, specific_impact)

                results.append({
                    'Feature_Set': feature_set,
                    'Motif': motif,
                    'N_Features': n_features,
                    'Specific_Impact': specific_impact,
                    'Random_Mean': random_mean,
                    'Random_Std': random_std,
                    'Z_Score': z_score,
                    'Percentile': percentile,
                    'P_Value': p_val,
                    'Significant': p_val < 0.05
                })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description='Run interpretability experiments with robust random controls (k=4)')
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n_random_trials', type=int, default=20, help='Number of random control trials')
    args = parser.parse_args()

    print("="*70)
    print("MOTIF-SPECIFIC ABLATION WITH ROBUST RANDOM CONTROLS (k=4)")
    print("="*70)
    print(f"SAE: latent_dim={args.latent_dim}, k={args.k}")
    print(f"Random trials: {args.n_random_trials}")

    # Load data from corrected metadata
    motif_features, non_sig_features = load_correlation_data()
    save_feature_mapping(motif_features)

    print(f"\nNon-significant features available: {len(non_sig_features)}")
    for motif_name, features in motif_features.items():
        print(f"{motif_name} significant features: {len(features)}")

    # Run motif-specific ablations
    print("\n" + "="*70)
    print("STEP 1: Motif-Specific Ablations")
    print("="*70)

    motif_specific_results = []

    for feature_set_name, features in motif_features.items():
        if len(features) == 0:
            print(f"\nSkipping {feature_set_name} (no significant features)")
            continue

        print(f"\nAblating {feature_set_name} features ({len(features)})...")
        experiment_name = f"ablate_{feature_set_name.lower().replace(' ', '_')}_k{args.k}"

        df = run_single_ablation(args.latent_dim, args.k, features, experiment_name)

        if df is not None:
            for motif in df['Motif'].unique():
                motif_data = df[df['Motif'] == motif]
                motif_specific_results.append({
                    'Feature_Set': feature_set_name,
                    'Motif': motif,
                    'N_Features': len(features),
                    'N': len(motif_data),
                    'Mean_Impact': motif_data['Ablation Impact'].mean(),
                    'Std_Impact': motif_data['Ablation Impact'].std(),
                    'SE_Impact': motif_data['Ablation Impact'].std() / np.sqrt(len(motif_data))
                })

    motif_specific_df = pd.DataFrame(motif_specific_results)
    motif_specific_df.to_csv(RESULTS_DIR / 'motif_specific_results.csv', index=False)

    # Run random controls for each unique feature count
    print("\n" + "="*70)
    print("STEP 2: Random Control Trials")
    print("="*70)

    unique_feature_counts = sorted(set(len(feats) for feats in motif_features.values() if len(feats) > 0))
    random_results_dict = {}

    for n_features in unique_feature_counts:
        random_df = run_random_control_trials(args.latent_dim, args.k, n_features,
                                              args.n_random_trials, non_sig_features)
        random_df.to_csv(RESULTS_DIR / f'random_{n_features}feat_trials_k{args.k}.csv', index=False)
        random_results_dict[n_features] = random_df

    # Statistical analysis
    print("\n" + "="*70)
    print("STEP 3: Statistical Comparison")
    print("="*70)

    combined_random = pd.concat(random_results_dict.values())
    sig_df = compute_statistical_significance(motif_specific_df, combined_random)
    sig_df.to_csv(RESULTS_DIR / 'statistical_tests.csv', index=False)

    print("\nStatistical Significance Results:")
    print(sig_df.to_string())

    # Generate plots
    print("\n" + "="*70)
    print("STEP 4: Generate Visualizations")
    print("="*70)

    plot_comparison_with_random_controls(motif_specific_df, random_results_dict, motif_features)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
