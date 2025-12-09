# Mixed-Motif Ablation Study Usage

## Overview

Added support for running ablation studies on **mixed-motif graphs** (graph IDs 4000-4999+) using **dominant motif labels** for classification.

## Key Changes

### 1. Modified Files

- **`run_interpretability_experiments.py`**: Added `--use_mixed_motifs` flag
- **`run_ablation.py`**: Added `--use_mixed_motifs` flag and logic to load mixed-motif graphs

### 2. How It Works

#### Graph Selection
- **Single-motif mode (default)**: Uses graphs 0-3999 from `outputs/test_graph_ids.json`
  - Activations from: `outputs/activations/layer2_new/test/`

- **Mixed-motif mode (`--use_mixed_motifs`)**: Uses ALL graphs ≥4000
  - Activations from: `outputs/activations/layer2/{train,test,val}/`
  - Searches all splits (train/test/val) since mixed graphs weren't used for GNN training

#### Motif Labeling
- Uses `get_dominant_motif(graph_id)` function (already existing in `run_ablation.py`)
- Determines dominant motif by counting which motif has the most participating nodes
- Maps to standard motif names: "Feedforward Loop", "Feedback Loop", "Single Input Module", "Cascade"

#### Analysis Pipeline
- Same as single-motif analysis:
  1. Ablate motif-specific features
  2. Run random control trials (ablating non-significant features)
  3. Compute statistical significance (Z-scores, percentiles, p-values)
  4. Generate comparison plots

### 3. Output Organization

Results are saved with `_mixed` suffix:
```
ablations/
├── interpretability_latent128_k8_rpb0.15_mixed_results/
│   ├── feature_motif_mapping.json
│   ├── motif_specific_results.csv
│   ├── random_13feat_trials.csv
│   ├── random_17feat_trials.csv
│   └── statistical_tests.csv
└── interpretability_latent128_k8_rpb0.15_mixed_plots/
    └── interpretability_vs_random_controls.png
```

## Usage Examples

### Step 1: Generate Mixed-Motif Activations (REQUIRED FIRST)

**IMPORTANT**: You must first generate activations for mixed-motif graphs using the current GNN checkpoint:

```bash
python generate_mixed_motif_activations.py
```

This will:
- Load the current GNN model from `checkpoints/gnn_model.pt`
- Process all mixed-motif graphs (4000-4999) from `virtual_graphs/data/all_graphs/raw_graphs/`
- Save 64-dim layer2 activations to `outputs/activations/layer2_new/mixed/`
- Take ~5-10 minutes for 1000 graphs

### Step 2: Run Mixed-Motif Ablation

After generating activations, run the ablation study:

```bash
python run_interpretability_experiments.py \
  --latent_dim 128 \
  --k 8 \
  --min_rpb 0.15 \
  --use_mixed_motifs
```

### With More Random Trials
```bash
python run_interpretability_experiments.py \
  --latent_dim 128 \
  --k 8 \
  --min_rpb 0.15 \
  --n_random_trials 100 \
  --use_mixed_motifs
```

### Compare Single vs Mixed
```bash
# Single-motif graphs
python run_interpretability_experiments.py \
  --latent_dim 128 --k 8 --min_rpb 0.15

# Mixed-motif graphs
python run_interpretability_experiments.py \
  --latent_dim 128 --k 8 --min_rpb 0.15 --use_mixed_motifs
```

## Key Differences: Single vs Mixed

| Aspect | Single-Motif | Mixed-Motif |
|--------|--------------|-------------|
| **Graph IDs** | 0-3999 | 4000-4999 |
| **Graphs used** | Test split only (~400) | All 1000 graphs |
| **Motif per graph** | Single motif type | Multiple motifs (dominant used) |
| **Activation path** | `layer2_new/test/` | `layer2_new/mixed/` |
| **GNN checkpoint** | Current (64-dim layer2) | Current (64-dim layer2) |
| **Metadata** | Template-based (accurate) | Structural detection (corrected) |
| **Use case** | Demonstrate monosemanticity | Test generalization to complex graphs |

## What This Tests

### Scientific Questions

1. **Generalization**: Do features learned from single-motif graphs generalize to mixed-motif graphs?
2. **Dominant motif hypothesis**: Is the dominant motif type the primary determinant of GNN predictions in mixed graphs?
3. **Feature robustness**: Are motif-specific features still causally relevant when multiple motifs coexist?

### Expected Outcomes

**If features generalize well:**
- Feedback Loop features should hurt graphs with dominant Feedback Loop motifs
- Impact should be weaker than single-motif case (dilution from other motifs)
- Z-scores should still be significant but lower magnitude

**If features don't generalize:**
- No clear relationship between feature set and dominant motif
- Similar impact across all dominant motif types
- Low Z-scores, not significantly different from random

## Data Details

### Mixed-Motif Graph Structure
- Graph 4500 example (from metadata):
  ```csv
  node_0: feedforward_loop=1, feedback_loop=1, single_input_module=1, cascade=1
  node_1: feedforward_loop=1, feedback_loop=1, single_input_module=1, cascade=1
  node_2: feedforward_loop=1, feedback_loop=1, single_input_module=0, cascade=1
  ...
  ```
- Each node can participate in multiple motifs
- Dominant motif = motif with highest sum across all nodes

### Activation Files
- Mixed-motif graphs have 64-dim activations (same as single-motif)
- **MUST be generated first** using `generate_mixed_motif_activations.py`
- Located in `outputs/activations/layer2_new/mixed/`
- Uses the SAME GNN checkpoint as single-motif graphs (`checkpoints/gnn_model.pt`)
- **DO NOT use** `layer2/` (contains activations from OLD GNN with different architecture)

## Visualization

The same plotting functions are used, generating:
- Bar charts comparing motif-specific vs random control impacts
- Grouped by dominant motif type on x-axis
- Error bars showing standard error across graphs

You can also use `plot_random_control_distributions.py`:
```bash
python plot_random_control_distributions.py \
  --results_dir ablations/interpretability_latent128_k8_rpb0.15_mixed_results \
  --output_dir ablations/interpretability_latent128_k8_rpb0.15_mixed_plots
```

## Notes

- The SAE features are trained on single-motif graphs only
- This is a **zero-shot transfer** test to mixed-motif graphs
- Dominant motif labeling may not capture full complexity of mixed graphs
- Alternative: Use multi-label classification or motif participation percentage
