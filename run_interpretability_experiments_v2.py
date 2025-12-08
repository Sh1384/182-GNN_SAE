#!/usr/bin/env python3
"""
Motif-Specific SAE Feature Ablation Experiments with Robust Random Controls

Runs ablation experiments to demonstrate SAE interpretability:
1. Ablate Feedback Loop features (39) → measure impact on all motif types
2. Ablate 100 random sets of 39 features → compute distribution of impacts
3. Ablate Single Input Module features (18) → measure impact on all motif types
4. Ablate 100 random sets of 18 features → compute distribution of impacts

Ablation Impact = MSE(GNN with ablated SAE) - MSE(GNN with full SAE)

Expected: Features significant for motif X should have larger impact on graphs
of type X compared to random feature sets.

Usage:
    python run_interpretability_experiments_v2.py --latent_dim 512 --k 16 --n_random_trials 100
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

# Directories
ABLATION_DIR = Path("ablations")
RESULTS_DIR = ABLATION_DIR / "interpretability_v2_results"
PLOTS_DIR = ABLATION_DIR / "interpretability_v2_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_feature_mapping():
    """Load feature-motif mapping from JSON."""
    with open('outputs/feature_motif_mapping.json', 'r') as f:
        return json.load(f)

def load_correlation_data():
    """Load correlation data to identify non-significant features."""
    df = pd.read_csv('outputs/latent_correlations.csv', index_col=0)

    # Get all non-significant features (any feature that is never significant)
    sig_features = df[df['significant_fdr'] == True]['feature'].unique()
    all_features = df['feature'].dropna().unique()
    non_sig_features = sorted([f for f in all_features if f not in sig_features])

    return non_sig_features

def run_single_ablation(latent_dim, k, features, experiment_name):
    """Run a single ablation experiment."""
    import sys

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

        experiment_name = f"ablate_random_trial_{n_features}feat_n{trial}"

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

def plot_comparison_with_random_controls(motif_specific_results, random_results_39, random_results_18):
    """Generate comprehensive comparison plots."""

    motif_colors = {
        'Feedforward Loop': '#377eb8',
        'Feedback Loop': '#ff7f00',
        'Single Input Module': '#4daf4a',
        'Cascade': '#e41a1c'
    }

    # Plot 1: Feedback Loop Features vs Random (39 features)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Feedback Loop comparison
    ax = axes[0]

    fbl_specific = motif_specific_results[motif_specific_results['Feature_Set'] == 'Feedback Loop']
    random_39 = random_results_39

    motif_order = ['Feedforward Loop', 'Feedback Loop', 'Single Input Module', 'Cascade']
    x_pos = np.arange(len(motif_order))
    width = 0.35

    # Specific features
    fbl_means = [fbl_specific[fbl_specific['Motif'] == m]['Mean_Impact'].values[0]
                 if len(fbl_specific[fbl_specific['Motif'] == m]) > 0 else 0
                 for m in motif_order]
    fbl_ses = [fbl_specific[fbl_specific['Motif'] == m]['SE_Impact'].values[0]
               if len(fbl_specific[fbl_specific['Motif'] == m]) > 0 else 0
               for m in motif_order]

    # Random controls
    random_means = [random_39[random_39['motif'] == m]['mean_impact'].mean()
                    if len(random_39[random_39['motif'] == m]) > 0 else 0
                    for m in motif_order]
    random_ses = [random_39[random_39['motif'] == m]['mean_impact'].std() / np.sqrt(len(random_39[random_39['motif'] == m]))
                  if len(random_39[random_39['motif'] == m]) > 0 else 0
                  for m in motif_order]

    bars1 = ax.bar(x_pos - width/2, fbl_means, width, label='Feedback Loop Features (39)',
                   yerr=fbl_ses, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5,
                   color=[motif_colors.get(m, '#999999') for m in motif_order])
    bars2 = ax.bar(x_pos + width/2, random_means, width, label='Random Controls (39 feat, 100 trials)',
                   yerr=random_ses, capsize=5, alpha=0.6, edgecolor='black', linewidth=1.5,
                   color='gray')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(motif_order, rotation=45, ha='right')
    ax.set_ylabel('Mean Ablation Impact\n(MSE increase)', fontsize=11)
    ax.set_title('Feedback Loop Features vs Random Controls\n(39 features ablated)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Right: Single Input Module comparison
    ax = axes[1]

    sim_specific = motif_specific_results[motif_specific_results['Feature_Set'] == 'Single Input Module']
    random_18 = random_results_18

    width = 0.35

    # Specific features
    sim_means = [sim_specific[sim_specific['Motif'] == m]['Mean_Impact'].values[0]
                 if len(sim_specific[sim_specific['Motif'] == m]) > 0 else 0
                 for m in motif_order]
    sim_ses = [sim_specific[sim_specific['Motif'] == m]['SE_Impact'].values[0]
               if len(sim_specific[sim_specific['Motif'] == m]) > 0 else 0
               for m in motif_order]

    # Random controls
    random_means_18 = [random_18[random_18['motif'] == m]['mean_impact'].mean()
                       if len(random_18[random_18['motif'] == m]) > 0 else 0
                       for m in motif_order]
    random_ses_18 = [random_18[random_18['motif'] == m]['mean_impact'].std() / np.sqrt(len(random_18[random_18['motif'] == m]))
                     if len(random_18[random_18['motif'] == m]) > 0 else 0
                     for m in motif_order]

    bars1 = ax.bar(x_pos - width/2, sim_means, width, label='Single Input Module Features (18)',
                   yerr=sim_ses, capsize=5, alpha=0.8, edgecolor='black', linewidth=1.5,
                   color=[motif_colors.get(m, '#999999') for m in motif_order])
    bars2 = ax.bar(x_pos + width/2, random_means_18, width, label='Random Controls (18 feat, 100 trials)',
                   yerr=random_ses_18, capsize=5, alpha=0.6, edgecolor='black', linewidth=1.5,
                   color='gray')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(motif_order, rotation=45, ha='right')
    ax.set_ylabel('Mean Ablation Impact\n(MSE increase)', fontsize=11)
    ax.set_title('Single Input Module Features vs Random Controls\n(18 features ablated)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('SAE Interpretability: Motif-Specific Features vs Random Controls',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'interpretability_vs_random_controls.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'interpretability_vs_random_controls.png'}")
    plt.close()

    # Plot 2: Distribution plots showing random trial variability
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, (motif, color) in enumerate(motif_colors.items()):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        # FBL specific
        fbl_val = fbl_specific[fbl_specific['Motif'] == motif]['Mean_Impact'].values
        fbl_val = fbl_val[0] if len(fbl_val) > 0 else 0

        # SIM specific
        sim_val = sim_specific[sim_specific['Motif'] == motif]['Mean_Impact'].values
        sim_val = sim_val[0] if len(sim_val) > 0 else 0

        # Random distributions
        random_39_vals = random_39[random_39['motif'] == motif]['mean_impact'].values
        random_18_vals = random_18[random_18['motif'] == motif]['mean_impact'].values

        # Plot histograms
        if len(random_39_vals) > 0:
            ax.hist(random_39_vals, bins=30, alpha=0.5, label='Random (39 feat)',
                   edgecolor='black', color='lightblue')
            ax.axvline(fbl_val, color='red', linewidth=3, linestyle='--',
                      label=f'FBL features ({fbl_val:.2e})')

        if len(random_18_vals) > 0:
            ax.hist(random_18_vals, bins=30, alpha=0.5, label='Random (18 feat)',
                   edgecolor='black', color='lightgreen')
            ax.axvline(sim_val, color='blue', linewidth=3, linestyle='--',
                      label=f'SIM features ({sim_val:.2e})')

        ax.set_xlabel('Mean Ablation Impact', fontsize=10)
        ax.set_ylabel('Frequency (across trials)', fontsize=10)
        ax.set_title(f'{motif} Graphs', fontsize=11, fontweight='bold', color=color)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Distribution of Random Control Impacts\n(Shows whether motif-specific features are outliers)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'random_control_distributions.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'random_control_distributions.png'}")
    plt.close()

def compute_statistical_significance(motif_specific_results, random_results):
    """Compute statistical tests comparing motif-specific to random."""
    results = []

    for feature_set in motif_specific_results['Feature_Set'].unique():
        specific_data = motif_specific_results[motif_specific_results['Feature_Set'] == feature_set]

        # Match random controls by number of features
        n_features = 39 if 'Feedback' in feature_set else 18
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
    parser = argparse.ArgumentParser(description='Run interpretability experiments with robust random controls')
    parser.add_argument('--latent_dim', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n_random_trials', type=int, default=100, help='Number of random control trials')
    args = parser.parse_args()

    print("="*70)
    print("MOTIF-SPECIFIC ABLATION WITH ROBUST RANDOM CONTROLS")
    print("="*70)
    print(f"SAE: latent_dim={args.latent_dim}, k={args.k}")
    print(f"Random trials: {args.n_random_trials}")

    # Load data
    feature_mapping = load_feature_mapping()
    non_sig_features = load_correlation_data()

    print(f"\nNon-significant features available: {len(non_sig_features)}")
    print(f"Feedback Loop significant features: {len(feature_mapping['Feedback Loop'])}")
    print(f"Single Input Module significant features: {len(feature_mapping['Single Input Module'])}")

    # Run motif-specific ablations
    print("\n" + "="*70)
    print("STEP 1: Motif-Specific Ablations")
    print("="*70)

    motif_specific_results = []

    for feature_set_name in ['Feedback Loop', 'Single Input Module']:
        features = feature_mapping[feature_set_name]
        if len(features) == 0:
            continue

        print(f"\nAblating {feature_set_name} features ({len(features)})...")
        experiment_name = f"ablate_{feature_set_name.lower().replace(' ', '_')}_v2"

        df = run_single_ablation(args.latent_dim, args.k, features, experiment_name)

        if df is not None:
            for motif in df['Motif'].unique():
                motif_data = df[df['Motif'] == motif]
                motif_specific_results.append({
                    'Feature_Set': feature_set_name,
                    'Motif': motif,
                    'N': len(motif_data),
                    'Mean_Impact': motif_data['Ablation Impact'].mean(),
                    'Std_Impact': motif_data['Ablation Impact'].std(),
                    'SE_Impact': motif_data['Ablation Impact'].std() / np.sqrt(len(motif_data))
                })

    motif_specific_df = pd.DataFrame(motif_specific_results)
    motif_specific_df.to_csv(RESULTS_DIR / 'motif_specific_results.csv', index=False)

    # Run random controls
    print("\n" + "="*70)
    print("STEP 2: Random Control Trials")
    print("="*70)

    random_39_df = run_random_control_trials(args.latent_dim, args.k, 39,
                                              args.n_random_trials, non_sig_features)
    random_39_df.to_csv(RESULTS_DIR / 'random_39feat_trials.csv', index=False)

    random_18_df = run_random_control_trials(args.latent_dim, args.k, 18,
                                              args.n_random_trials, non_sig_features)
    random_18_df.to_csv(RESULTS_DIR / 'random_18feat_trials.csv', index=False)

    # Statistical analysis
    print("\n" + "="*70)
    print("STEP 3: Statistical Comparison")
    print("="*70)

    combined_random = pd.concat([random_39_df, random_18_df])
    sig_df = compute_statistical_significance(motif_specific_df, combined_random)
    sig_df.to_csv(RESULTS_DIR / 'statistical_tests.csv', index=False)

    print("\nStatistical Significance Results:")
    print(sig_df.to_string())

    # Generate plots
    print("\n" + "="*70)
    print("STEP 4: Generate Visualizations")
    print("="*70)

    plot_comparison_with_random_controls(motif_specific_df, random_39_df, random_18_df)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Plots: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
