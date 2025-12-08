#!/usr/bin/env python3
"""
Motif-Specific SAE Feature Ablation Experiments

Runs 3 ablation experiments to demonstrate SAE interpretability:
1. Ablate Feedback Loop features → measure impact on all motif types
2. Ablate Single Input Module features → measure impact on all motif types
3. Ablate Random Control features → measure impact on all motif types

Expected: Features significant for motif X should have largest impact on graphs of type X.

Usage:
    python run_interpretability_experiments.py --latent_dim 512 --k 16
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

# Directories
ABLATION_DIR = Path("ablations")
RESULTS_DIR = ABLATION_DIR / "interpretability_results"
PLOTS_DIR = ABLATION_DIR / "interpretability_plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_feature_mapping():
    """Load feature-motif mapping from JSON."""
    with open('outputs/feature_motif_mapping.json', 'r') as f:
        return json.load(f)

def run_ablation_experiment(latent_dim, k, feature_set_name, features):
    """Run a single ablation experiment for a specific feature set."""
    if len(features) == 0:
        print(f"Skipping {feature_set_name} - no features to ablate")
        return None

    feature_str = ','.join(features)
    experiment_name = f"ablate_{feature_set_name.lower().replace(' ', '_')}"

    print(f"\n{'='*60}")
    print(f"Running: {feature_set_name} Ablation")
    print(f"{'='*60}")
    print(f"Ablating {len(features)} features: {features[:5]}...")

    # Run ablation via run_ablation.py
    import sys
    cmd = [
        sys.executable, "run_ablation.py",
        "--latent_dim", str(latent_dim),
        "--k", str(k),
        "--feature", feature_str,
        "--experiment_name", experiment_name
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running ablation for {feature_set_name}:")
        print(result.stderr)
        return None

    print(result.stdout)

    # Load results
    results_file = ABLATION_DIR / "results" / f"{experiment_name}_results.csv"
    if not results_file.exists():
        print(f"Warning: Results file not found: {results_file}")
        return None

    df = pd.read_csv(results_file)
    df['Ablated_Feature_Set'] = feature_set_name

    return df

def compute_statistics(all_results):
    """Compute statistical comparisons across ablation experiments."""
    stats_results = []

    for ablated_set in all_results['Ablated_Feature_Set'].unique():
        ablated_data = all_results[all_results['Ablated_Feature_Set'] == ablated_set]

        for motif in ablated_data['Motif'].unique():
            motif_data = ablated_data[ablated_data['Motif'] == motif]

            if len(motif_data) > 0:
                mean_impact = motif_data['Ablation Impact'].mean()
                std_impact = motif_data['Ablation Impact'].std()
                median_impact = motif_data['Ablation Impact'].median()

                stats_results.append({
                    'Ablated_Feature_Set': ablated_set,
                    'Affected_Motif': motif,
                    'N': len(motif_data),
                    'Mean_Impact': mean_impact,
                    'Std_Impact': std_impact,
                    'Median_Impact': median_impact,
                    'SE_Impact': std_impact / np.sqrt(len(motif_data)) if len(motif_data) > 0 else 0
                })

    return pd.DataFrame(stats_results)

def plot_comparative_results(stats_df, all_results):
    """Generate comprehensive comparative plots."""

    # Define motif order and colors
    motif_order = ['Feedforward Loop', 'Feedback Loop', 'Single Input Module', 'Cascade']
    motif_colors = {
        'Feedforward Loop': '#377eb8',
        'Feedback Loop': '#ff7f00',
        'Single Input Module': '#4daf4a',
        'Cascade': '#e41a1c'
    }

    # Filter to keep only existing motifs
    motif_order = [m for m in motif_order if m in stats_df['Affected_Motif'].unique()]

    # Plot 1: Heatmap of Ablation Impact
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    pivot_data = stats_df.pivot(index='Ablated_Feature_Set',
                                  columns='Affected_Motif',
                                  values='Mean_Impact')

    # Reorder columns to match motif_order
    pivot_data = pivot_data[[col for col in motif_order if col in pivot_data.columns]]

    sns.heatmap(pivot_data, annot=True, fmt='.2e', cmap='RdYlBu_r',
                center=0, cbar_kws={'label': 'Mean Ablation Impact (MSE increase)'},
                ax=ax, linewidths=0.5, linecolor='gray')

    ax.set_title('SAE Interpretability: Motif-Specific Feature Ablation Impact\n' +
                 'Rows = Ablated Features | Columns = Affected Graph Type',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Affected Motif Type (Graph Type)', fontsize=11)
    ax.set_ylabel('Ablated Feature Set', fontsize=11)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'interpretability_heatmap.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'interpretability_heatmap.png'}")
    plt.close()

    # Plot 2: Bar Chart with Error Bars
    fig, axes = plt.subplots(1, len(stats_df['Ablated_Feature_Set'].unique()),
                             figsize=(16, 5), sharey=True)

    if len(stats_df['Ablated_Feature_Set'].unique()) == 1:
        axes = [axes]

    for idx, ablated_set in enumerate(stats_df['Ablated_Feature_Set'].unique()):
        ax = axes[idx]
        subset = stats_df[stats_df['Ablated_Feature_Set'] == ablated_set]
        subset = subset.set_index('Affected_Motif').reindex(motif_order).reset_index()

        colors = [motif_colors.get(m, '#999999') for m in subset['Affected_Motif']]

        ax.bar(range(len(subset)), subset['Mean_Impact'],
               yerr=subset['SE_Impact'],
               color=colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.5)

        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset['Affected_Motif'], rotation=45, ha='right')
        ax.set_title(f'Ablating:\n{ablated_set}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if idx == 0:
            ax.set_ylabel('Mean Ablation Impact\n(GNN MSE Increase)', fontsize=10)

    plt.suptitle('Motif-Specific Ablation Impact Across Graph Types',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'interpretability_bar_chart.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'interpretability_bar_chart.png'}")
    plt.close()

    # Plot 3: Boxplots for each ablation experiment
    fig, axes = plt.subplots(1, len(all_results['Ablated_Feature_Set'].unique()),
                             figsize=(18, 5), sharey=True)

    if len(all_results['Ablated_Feature_Set'].unique()) == 1:
        axes = [axes]

    for idx, ablated_set in enumerate(all_results['Ablated_Feature_Set'].unique()):
        ax = axes[idx]
        subset = all_results[all_results['Ablated_Feature_Set'] == ablated_set]

        # Prepare data for boxplot
        box_data = [subset[subset['Motif'] == m]['Ablation Impact'].values
                    for m in motif_order if m in subset['Motif'].unique()]
        box_labels = [m for m in motif_order if m in subset['Motif'].unique()]
        box_colors = [motif_colors[m] for m in box_labels]

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        boxprops=dict(linewidth=2, edgecolor='black'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2.5, color='red'),
                        showfliers=False)

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticklabels(box_labels, rotation=45, ha='right')
        ax.set_title(f'Ablating: {ablated_set}', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        if idx == 0:
            ax.set_ylabel('Ablation Impact (GNN MSE Increase)', fontsize=10)

    plt.suptitle('Distribution of Ablation Impact by Motif Type',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'interpretability_boxplots.png', dpi=200, bbox_inches='tight')
    print(f"Saved: {PLOTS_DIR / 'interpretability_boxplots.png'}")
    plt.close()

def print_interpretability_summary(stats_df):
    """Print summary showing evidence of interpretability."""
    print("\n" + "="*70)
    print("INTERPRETABILITY ANALYSIS SUMMARY")
    print("="*70)

    # For each ablated feature set, find which motif type is most affected
    for ablated_set in stats_df['Ablated_Feature_Set'].unique():
        subset = stats_df[stats_df['Ablated_Feature_Set'] == ablated_set]
        max_impact = subset.loc[subset['Mean_Impact'].idxmax()]

        print(f"\n{ablated_set} Features:")
        print(f"  Most affected motif: {max_impact['Affected_Motif']}")
        print(f"  Mean impact: {max_impact['Mean_Impact']:.2e}")

        print(f"\n  Impact by motif type:")
        for _, row in subset.sort_values('Mean_Impact', ascending=False).iterrows():
            print(f"    {row['Affected_Motif']}: {row['Mean_Impact']:.2e} ± {row['SE_Impact']:.2e}")

    # Interpretability check
    print("\n" + "="*70)
    print("INTERPRETABILITY CHECK:")
    print("="*70)

    feedback_ablation = stats_df[stats_df['Ablated_Feature_Set'] == 'Feedback Loop']
    sim_ablation = stats_df[stats_df['Ablated_Feature_Set'] == 'Single Input Module']

    if len(feedback_ablation) > 0:
        fbl_on_fbl = feedback_ablation[feedback_ablation['Affected_Motif'] == 'Feedback Loop']['Mean_Impact'].values
        fbl_on_ffl = feedback_ablation[feedback_ablation['Affected_Motif'] == 'Feedforward Loop']['Mean_Impact'].values

        if len(fbl_on_fbl) > 0 and len(fbl_on_ffl) > 0:
            ratio = fbl_on_fbl[0] / fbl_on_ffl[0] if fbl_on_ffl[0] > 0 else float('inf')
            check1 = "✓ PASS" if ratio > 1.0 else "✗ FAIL"
            print(f"\nFeedback Loop features affect Feedback Loop graphs {ratio:.2f}x more than Feedforward Loop graphs: {check1}")

    if len(sim_ablation) > 0:
        sim_on_sim = sim_ablation[sim_ablation['Affected_Motif'] == 'Single Input Module']['Mean_Impact'].values
        sim_on_ffl = sim_ablation[sim_ablation['Affected_Motif'] == 'Feedforward Loop']['Mean_Impact'].values

        if len(sim_on_sim) > 0 and len(sim_on_ffl) > 0:
            ratio = sim_on_sim[0] / sim_on_ffl[0] if sim_on_ffl[0] > 0 else float('inf')
            check2 = "✓ PASS" if ratio > 1.0 else "✗ FAIL"
            print(f"Single Input Module features affect SIM graphs {ratio:.2f}x more than Feedforward Loop graphs: {check2}")

    print("\n" + "="*70)

def main():
    parser = argparse.ArgumentParser(description='Run motif-specific ablation experiments')
    parser.add_argument('--latent_dim', type=int, required=True, help='SAE latent dimension')
    parser.add_argument('--k', type=int, required=True, help='SAE top-k sparsity')
    args = parser.parse_args()

    print("="*70)
    print("MOTIF-SPECIFIC FEATURE ABLATION EXPERIMENTS")
    print("="*70)
    print(f"SAE Configuration: latent_dim={args.latent_dim}, k={args.k}")

    # Load feature mapping
    feature_mapping = load_feature_mapping()
    print(f"\nFeature sets:")
    for name, features in feature_mapping.items():
        print(f"  {name}: {len(features)} features")

    # Run ablation experiments
    all_results = []

    for feature_set_name in ['Feedback Loop', 'Single Input Module', 'Random Control']:
        features = feature_mapping.get(feature_set_name, [])

        if len(features) > 0:
            df = run_ablation_experiment(args.latent_dim, args.k, feature_set_name, features)
            if df is not None:
                all_results.append(df)

    if len(all_results) == 0:
        print("Error: No ablation experiments completed successfully")
        return

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(RESULTS_DIR / 'all_ablation_results.csv', index=False)
    print(f"\nSaved combined results to: {RESULTS_DIR / 'all_ablation_results.csv'}")

    # Compute statistics
    stats_df = compute_statistics(combined_results)
    stats_df.to_csv(RESULTS_DIR / 'ablation_statistics.csv', index=False)
    print(f"Saved statistics to: {RESULTS_DIR / 'ablation_statistics.csv'}")

    # Generate plots
    print("\nGenerating comparative visualizations...")
    plot_comparative_results(stats_df, combined_results)

    # Print summary
    print_interpretability_summary(stats_df)

    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")

if __name__ == "__main__":
    main()
