# Quick Start: Mixed-Motif Ablation Study

## Problem Identified

The `outputs/activations/layer2/` folder contains activations from an **OLD GNN** with different architecture (hidden_dim1=248 vs current=88). We cannot use these for ablation studies with the current SAE.

## Solution

Generate fresh activations for mixed-motif graphs using the **CURRENT** GNN checkpoint.

## Setup Steps

### 1. Generate Activations (ONE TIME)

```bash
cd 182-GNN_SAE
python generate_mixed_motif_activations.py
```

**What this does:**
- Loads GNN from `checkpoints/gnn_model.pt` (88→64→1 architecture)
- Processes 1000 mixed-motif graphs (4000-4999)
- Saves layer2 activations (64-dim) to `outputs/activations/layer2_new/mixed/`
- Takes ~5-10 minutes

**Output:**
```
Found 1000 mixed-motif graphs
Range: graph_4000 to graph_4999
Successfully saved: 1000
Activations saved to: outputs/activations/layer2_new/mixed/
```

### 2. Run Ablation Study

```bash
python run_interpretability_experiments.py \
  --latent_dim 128 \
  --k 8 \
  --min_rpb 0.15 \
  --use_mixed_motifs \
  --n_random_trials 100
```

## File Structure

### Before:
```
outputs/activations/
├── layer2/              # OLD GNN (248→64→1) - DON'T USE
│   ├── train/           # Contains graph_4000-4999
│   ├── test/
│   └── val/
└── layer2_new/          # CURRENT GNN (88→64→1)
    ├── train/           # Single-motif: 0-3999
    ├── test/
    └── val/
```

### After running generate_mixed_motif_activations.py:
```
outputs/activations/
├── layer2/              # OLD GNN - DON'T USE
│   └── ...
└── layer2_new/          # CURRENT GNN
    ├── train/           # Single-motif: 0-3999
    ├── test/
    ├── val/
    └── mixed/           # Mixed-motif: 4000-4999 ✓ NEW!
        ├── graph_4000.pt
        ├── graph_4001.pt
        └── ...
```

## Why This Matters

1. **Architecture consistency**: SAE was trained on 64-dim activations from current GNN
2. **Correct predictions**: Mixed graphs need predictions from current GNN for ablation impact calculation
3. **Fair comparison**: Single vs mixed comparisons require same GNN architecture

## Verification

Check that activations were generated correctly:

```bash
# Count files
ls outputs/activations/layer2_new/mixed/ | wc -l
# Should show: 1000

# Check sample shape
python -c "import torch; print(torch.load('outputs/activations/layer2_new/mixed/graph_4500.pt').shape)"
# Should show: torch.Size([10, 64])
```

## Files Modified

1. **`generate_mixed_motif_activations.py`** (NEW)
   - Loads current GNN checkpoint
   - Generates activations for graphs 4000-4999
   - Saves to `layer2_new/mixed/`

2. **`run_ablation.py`**
   - Updated to look in `layer2_new/mixed/` for mixed-motif graphs
   - Shows error message if activations not found
   - Suggests running `generate_mixed_motif_activations.py`

3. **`run_interpretability_experiments.py`**
   - No changes needed for activation generation
   - `--use_mixed_motifs` flag passes through to `run_ablation.py`

## Troubleshooting

### Error: "Mixed-motif activations not found"

**Solution**: Run `python generate_mixed_motif_activations.py` first

### Error: "No mixed-motif graphs found in raw_graphs/"

**Solution**: Verify graphs exist:
```bash
ls virtual_graphs/data/all_graphs/raw_graphs/graph_4*.pkl | wc -l
# Should show 1000
```

### Error: "GNN checkpoint not found"

**Solution**: Verify checkpoint exists:
```bash
ls -lh checkpoints/gnn_model.pt
```

## Next Steps

After generating activations:

1. Run ablation study on mixed-motif graphs
2. Compare with single-motif results
3. Use `plot_random_control_distributions.py` to visualize

```bash
# After ablation completes:
python plot_random_control_distributions.py \
  --results_dir ablations/interpretability_latent128_k8_rpb0.15_mixed_results \
  --output_dir ablations/interpretability_latent128_k8_rpb0.15_mixed_plots
```
