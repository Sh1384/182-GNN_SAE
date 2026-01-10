# METHODOLOGY (Poster Version)

## Visual Pipeline Flow
```
[Synthetic Graphs] → [GNN Training] → [SAE Decomposition] → [Interpretability Analysis]
```

---

## 1. Synthetic Data Generation
**Generate 5,000 graphs with known motif structure**

- **4 Motif Types**: Feedforward Loop (FFL), Feedback Loop (FB), Single-Input Module (SIM), Cascade
- **Graph Properties**: 10 nodes, directed weighted edges, U[-1,1] weights
- **Expression Simulation**: Leaky dynamical system over 50 timesteps
  ```
  x_{t+1} = (1-γ)x_t + γ·σ(Wx_t) + ε
  ```
- **Data Split**: 4,000 single-motif + 1,000 mixed-motif graphs

---

## 2. GNN Training on Node Imputation
**Train GNNs to predict masked node expressions**

- **Architectures**: GCN (3 layers: 2→88→64→1) & GAT (multi-head attention)
- **Task**: Mask 20% of nodes, predict their expression values (MSE loss)
- **Data Split**: 80/10/10 train/val/test, stratified by motif
- **Key Output**: Extract layer 2 activations (64-dim) for all graphs
- **Performance**: Test MSE = 0.0031 (GCN), outperforms MLP baseline

---

## 3. Sparse Autoencoder (SAE) Decomposition
**Decompose frozen GNN activations into sparse features**

- **Architecture**: 64 (GNN activations) → 512 (sparse latent) → 64 (reconstruction)
- **Sparsity**: Top-K selection (k=8, ~1.5% active features)
- **Training**: Minimize reconstruction loss on frozen activations
- **Hypothesis**: Sparse latent features should act as motif detectors

---

## 4. Interpretability Analysis
**Link SAE features to biological motifs**

**A. Correlation Analysis**
- **Point-biserial correlation**: Measure association between feature activation & motif presence
- **Statistical Testing**: Permutation test (1,000 permutations) + FDR correction (α=0.05)
- **Feature Selection**: High correlation with target motif (|r|>0.5), low with others (|r|<0.2)

**B. Causal Validation**
- **Ablation Experiment**: Zero out specific SAE features in latent space
- **Measurement**: Impact on GNN prediction accuracy per motif type
- **Control**: Compare to random feature ablation baseline

---

## Statistical Benchmarking
**Rigorous multi-seed evaluation**

- **20 seeds** per model to account for training stochasticity
- **Wilcoxon signed-rank test** for paired model comparison
- **Bootstrap 95% CI** (10,000 resamples) for uncertainty quantification
- **Sensitivity analysis**: Validate hyperparameter choices (timesteps, noise)

---

## Key Design Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Graph size | 10 nodes | Consistent comparison, computational efficiency |
| Masking rate | 20% | Balance between supervision and difficulty |
| Sparsity | Top-8/512 (~1.5%) | High sparsity → interpretable features |
| Timesteps | 50 | Sufficient for convergence (validated) |
| SAE expansion | 8× (64→512) | Standard for feature discovery |

---

## VISUAL SUGGESTIONS FOR POSTER

### Panel 1: Data Generation
- Show 4 motif diagrams (node-edge graphs)
- Include expression heatmap (nodes × time)

### Panel 2: GNN Architecture
- Neural network diagram with layer dimensions
- Highlight layer 2 (activation extraction point)

### Panel 3: SAE Decomposition
- Encoder-decoder schematic
- Visualization of sparse vs dense activations

### Panel 4: Interpretability
- Correlation heatmap (features × motifs)
- Ablation bar plot (MSE change by motif)

---

## CONDENSED TEXT VERSION (For tight space)

### Methods
We generated 5,000 synthetic gene regulatory networks (10 nodes each) containing four motif types: Feedforward Loops, Feedback Loops, Single-Input Modules, and Cascades. Expression dynamics were simulated over 50 timesteps using a leaky dynamical system.

Graph Convolutional Networks (GCN, 3 layers) and Graph Attention Networks (GAT) were trained on node imputation: predicting 20% masked node expressions (MSE loss). Layer 2 activations (64-dim) were extracted and frozen.

Sparse Autoencoders (64→512→64) with Top-K sparsity (k=8) decomposed GNN activations into interpretable features. Point-biserial correlation with permutation testing (FDR-corrected, α=0.05) identified motif-specific features. Causal ablation experiments validated functional relevance by zeroing features and measuring prediction impact.

Multi-seed benchmarking (20 seeds) with Wilcoxon tests confirmed GNN superiority over MLP and mean baselines (p<0.001).

---

## ULTRA-CONDENSED VERSION (For minimal space)

**Data:** 5,000 synthetic 10-node graphs with 4 motif types; expression simulated over 50 timesteps.

**GNN Training:** GCN/GAT trained on node imputation (mask 20%, predict expression); layer 2 activations (64-dim) extracted.

**SAE Analysis:** Sparse autoencoder (64→512→64, Top-8 sparsity) decomposes activations into interpretable features.

**Interpretability:** Point-biserial correlation + permutation testing identifies motif-specific features; causal ablation validates functional relevance.

**Evaluation:** Multi-seed benchmarking (n=20) with Wilcoxon tests; GCN achieves MSE=0.0031, significantly outperforming baselines.
