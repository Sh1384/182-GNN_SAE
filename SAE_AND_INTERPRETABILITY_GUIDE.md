# Sparse Autoencoder (SAE) and Interpretability Analysis Guide

## Overview

This document provides a thorough explanation of the Sparse Autoencoder (SAE) and interpretability analysis pipeline in this project. The goal is to discover interpretable features in GNN layer activations that correlate with specific network motifs.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Sparse Autoencoder (SAE) Explained](#sparse-autoencoder-sae-explained)
3. [Interpretability Analysis](#interpretability-analysis)
4. [Workflow: From GNN to Interpretable Features](#workflow-from-gnn-to-interpretable-features)
5. [Key Concepts & Metrics](#key-concepts--metrics)
6. [Code Walkthrough](#code-walkthrough)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Complete Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. GNN Training (gnn_train.py)                                 │
│     ├─ Train GCN/GAT model on synthetic graphs                 │
│     ├─ Extract layer activations during forward passes         │
│     └─ Save activations: outputs/activations/layer{1,2}/{split}│
│                                                                  │
│  2. Sparse Autoencoder Training (sparse_autoencoder.py)         │
│     ├─ Load frozen GNN activations                             │
│     ├─ Train SAE with sparsity constraint                      │
│     ├─ Discover latent features in compressed space           │
│     └─ Save latent features: outputs/sae_latents/layer{1,2}/   │
│                                                                  │
│  3. Interpretability Analysis (interpretability_analysis.py)    │
│     ├─ Load SAE latent features                                │
│     ├─ Compute feature-motif correlations                      │
│     ├─ Identify interpretable features                         │
│     └─ Generate visualizations & metrics                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Sparse Autoencoder (SAE) Explained

### What is a Sparse Autoencoder?

A Sparse Autoencoder is a neural network that learns a compressed representation (latent space) of high-dimensional data while enforcing sparsity—meaning most latent units should be inactive (close to zero) at any given time.

### Architecture

```
Input Activations (from GNN)
           |
           v
    Encoder (Linear Layer)
           |
           v
        ReLU Activation
           |
           v
    Latent Representation (Z)  ← Sparse! Most values ~0
           |
           v
    Decoder (Linear Layer)
           |
           v
    Reconstructed Activations
```

### Mathematical Formulation

**Loss Function:**
```
L_total = L_reconstruction + λ * L_sparsity

L_reconstruction = ||x - x̂||²_2    (Mean Squared Error)
```

**Sparsity Penalties:**

1. **L1 Penalty** (default):
   ```
   L_sparsity = E[|z|]  = mean(|latent_activations|)
   ```
   - Simple: penalizes magnitude of latent activations
   - Encourages many dimensions to be exactly 0

2. **Top-K Penalty**:
   ```
   L_sparsity = mean(Σ|z_i| for i not in top-k)
   ```
   - Only the top-k most active features per sample are kept
   - All other features are penalized
   - More aggressive sparsity control

### Key Parameters

| Parameter | Purpose | Value |
|-----------|---------|-------|
| `input_dim` | Dimension of GNN layer activations | Layer 1: 64, Layer 2: 1 |
| `latent_dim` | Dimension of compressed space | Layer 1: 512, Layer 2: 32 |
| `sparsity_lambda` | Weight of sparsity penalty | 1e-3 |
| `sparsity_type` | Type of sparsity regularization | 'l1' or 'topk' |
| `topk_ratio` | Fraction of dimensions to keep (topk only) | 0.1 (keep 10%) |

### Training Details

From `sparse_autoencoder.py:362-464`:

1. **Load Activations**: Read frozen GNN layer activations from disk
   ```python
   train_activations = load_activations(activation_dir, "train")  # Shape: (num_nodes, input_dim)
   ```

2. **Create DataLoader**: Batch activations for efficient training
   ```python
   train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
   ```

3. **Training Loop**:
   ```python
   for epoch in range(num_epochs):
       x_hat, z = model(x)  # Forward pass
       total_loss, recon_loss, sparsity_loss = model.loss(x, x_hat, z)
       total_loss.backward()
       optimizer.step()
   ```

4. **Early Stopping**: Monitor validation loss, stop if no improvement for 15 epochs

5. **Extract Latents**: After training, encode all activations to latent space
   ```python
   z = model.encode(x)  # Extract latent features
   ```

### Why Sparsity Matters

**Benefits:**
- **Interpretability**: Fewer active features = easier to understand
- **Efficiency**: Sparse representations are computationally efficient
- **Feature Discovery**: Forces the model to find the most essential features
- **Disentanglement**: Different features can specialize in different patterns

**Example:**
- Without sparsity: All 512 latent dimensions might be used for every input
- With sparsity: Only ~50 dimensions might be active, making it clear which features matter

---

## Interpretability Analysis

### Goal

Find SAE latent features that:
1. Correlate strongly with specific motifs
2. Are largely independent from other motifs
3. Can be visualized and explained

### Three Main Analysis Techniques

#### 1. Point-Biserial Correlation

**What it does**: Measures correlation between a continuous variable (SAE feature) and a binary variable (motif presence).

**Formula**:
```
r_pb = (M1 - M0) / sqrt(σ²) * sqrt(n1*n0 / n²)

where:
  M1 = mean of feature for motif-present nodes
  M0 = mean of feature for motif-absent nodes
  n1, n0 = counts of each group
```

**In the code** (`interpretability_analysis.py:129-173`):
```python
# Create binary label: 1 if motif present, 0 otherwise
binary_labels = (motif_labels == motif_idx).astype(int)

# For each feature, compute correlation with binary label
for feature_idx in range(n_features):
    corr, pval = pointbiserialr(binary_labels, feature_values)
    # Results stored: {motif, feature_idx, correlation, p_value}
```

**Interpretation**:
- r = 1: feature always high when motif present, low otherwise (perfect correlation)
- r = 0: no relationship between feature and motif
- r = -1: feature always low when motif present (negative correlation)

**Advantages**:
- Fast to compute
- Straightforward interpretation
- Statistical significance (p-value)

#### 2. Mutual Information (MI)

**What it does**: Measures how much information a feature contains about the motif type (discrete).

**Formula**:
```
I(Z; M) = Σ P(z, m) * log(P(z, m) / (P(z) * P(m)))

where:
  Z = discretized feature values
  M = motif labels
```

**In the code** (`interpretability_analysis.py:175-224`):
```python
# Discretize continuous features into bins
feature_discrete = np.digitize(feature_values, bins=np.linspace(...))

# Compute mutual information with motif labels
mi = mutual_info_score(valid_labels, feature_discrete)
```

**Interpretation**:
- MI = 0: feature is independent of motif type
- Higher MI: feature better predicts motif type
- No upper limit (depends on dimensionality)

**Advantages**:
- Captures non-linear relationships
- Works with multi-class (4 motifs) not just binary
- Model-agnostic

#### 3. Feature Selection Criteria

**Interpretable Feature Definition** (`interpretability_analysis.py:226-277`):

A feature is considered "interpretable" if:
1. **High correlation to one motif**: `max_correlation >= threshold_high` (default 0.5)
2. **Low correlation to others**: `max_other_correlation < threshold_low` (default 0.2)

**Logic**:
```python
for feature_idx in features:
    max_corr = max(|correlations with all motifs|)

    if max_corr >= 0.5:  # High correlation to something
        target_motif = motif with max correlation
        other_corrs = correlations with other motifs

        if max(other_corrs) < 0.2:  # Low to others
            → Feature is interpretable!
```

**Example**:
```
Feature 42:
  - Feedforward loop:    corr = 0.72  ✓
  - Feedback loop:       corr = 0.15  ✓
  - Single input module: corr = 0.08  ✓
  - Cascade:             corr = 0.05  ✓

Result: INTERPRETABLE (target: feedforward_loop)
```

### Visualization: Motif-Feature Heatmap

Shows correlations between all features and all motifs.

```
        Feature_0  Feature_1  Feature_2  ...  Feature_512
Cascade     0.05      -0.12      0.68           0.02
Feedback    0.15       0.71     -0.05           0.14
Feedforward 0.72      -0.08      0.12          -0.04
SingleInput 0.08       0.05     -0.09           0.78
```

Interpretation:
- Dark red = high positive correlation
- Dark blue = high negative correlation
- White = no correlation

---

## Workflow: From GNN to Interpretable Features

### Step-by-Step Process

#### Step 1: Train GNN (gnn_train.py)

```python
# Train GCN/GAT model
model = GCNModel(...)
trainer = GNNTrainer(model, ...)

for epoch in range(NUM_EPOCHS):
    train_loss = trainer.train_epoch(train_loader)
    val_loss = trainer.validate(val_loader)

# Extract activations during inference
trainer.extract_and_save_activations(train_loader, "outputs", "train")
# Saves: outputs/activations/layer1/train/graph_0.pt, graph_1.pt, ...
#        outputs/activations/layer2/train/graph_0.pt, graph_1.pt, ...
```

**Outputs:**
- GNN model weights: `checkpoints/gnn_model.pt`
- Layer 1 activations: `outputs/activations/layer1/{train,val,test}/graph_*.pt`
  - Shape: (num_nodes_in_graph, 64)
- Layer 2 activations: `outputs/activations/layer2/{train,val,test}/graph_*.pt`
  - Shape: (num_nodes_in_graph, 1)

#### Step 2: Train SAE (sparse_autoencoder.py)

```python
# Load all activations into single tensor
train_activations = load_activations(activation_dir, "train")  # Shape: (total_nodes, 64)

# Create SAE: 64-dim → 512-dim (expansion then sparsity)
model = SparseAutoencoder(input_dim=64, latent_dim=512, sparsity_lambda=1e-3)

# Train with combined loss
for epoch in range(100):
    x_hat, z = model(x)
    total_loss = ||x - x̂||² + 1e-3 * |z|
    total_loss.backward()

# Extract latent features
z_train = model.encode(train_activations)  # Shape: (total_nodes, 512)
```

**Why expand then compress?**
- Expansion (64 → 512) allows learning more features
- Sparsity forces most to be zero
- Net effect: selective feature discovery

**Outputs:**
- SAE model weights: `checkpoints/sae_layer1.pt`
- Latent features: `outputs/sae_latents/layer1/{train,val,test}/`
  - `all_latents.pt`: (total_nodes, 512)
  - `node_0.pt`, `node_1.pt`, ...: individual node latents

#### Step 3: Interpretability Analysis (interpretability_analysis.py)

```python
# Load SAE latents
latents = torch.load("outputs/sae_latents/layer1/train/all_latents.pt")  # (nodes, 512)

# Create node-level motif labels
motif_labels, graph_ids = analyzer.create_node_motif_labels()  # (nodes,)

# Compute correlations
corr_df = analyzer.compute_pointbiserial_correlations(latents, motif_labels)
# Result: DataFrame with columns [motif, feature_idx, correlation, p_value, layer]
# Example rows:
#   motif='cascade',     feature_idx=0,   correlation=0.05,   p_value=0.42
#   motif='feedback',    feature_idx=0,   correlation=0.15,   p_value=0.01
#   motif='feedforward', feature_idx=0,   correlation=0.72,   p_value=0.00

# Identify interpretable features
interp_df = analyzer.identify_interpretable_features(corr_df)
# Result: DataFrame with columns [layer, feature_idx, target_motif,
#                                  target_correlation, max_other_correlation]
# Example rows:
#   layer='layer1', feature_idx=0, target_motif='feedforward',
#   target_correlation=0.72, max_other_correlation=0.15
```

**Outputs:**
- Correlations: `outputs/interpretability/motif_feature_correlation_layer1.csv`
- Mutual info: `outputs/interpretability/motif_feature_mutualinfo_layer1.csv`
- Interpretable features: `outputs/interpretability/interpretable_features_layer1.csv`
- Visualizations: `outputs/interpretability/heatmap_layer1.png`

---

## Key Concepts & Metrics

### Reconstruction Loss

Measures how well the SAE can reconstruct original activations:
```
L_recon = mean((x - x̂)²)
```

**Interpretation:**
- Lower is better
- Low recon loss with high sparsity = good compression without information loss
- High recon loss = SAE struggles to capture features with sparsity constraint

### Sparsity Loss

Measures how sparse the latent representation is:
```
L_sparse = mean(|z|)   (L1 penalty)
```

**Interpretation:**
- Higher = more sparse (most latents near 0)
- Lower = less sparse (more latents active)
- Trade-off: increasing sparsity increases reconstruction error

### Total Loss Trade-off

```
L_total = L_recon + λ * L_sparse
```

The hyperparameter λ controls this trade-off:
- λ = 0: no sparsity constraint (standard autoencoder)
- λ = 1e-3: balanced compression and reconstruction
- λ = 0.1: strong sparsity emphasis

### Feature Activations

For each node, the SAE produces a 512-dim latent vector z.

**Example interpretation:**
```
Node 1234 (in feedforward_loop graph):
  z = [0.05, 0.00, 0.82, 0.00, 0.01, 0.00, 0.00, ...]
        └─ inactive  └─ active   └─ inactive

Only 3 dimensions are active!
The strongest (0.82) could be a "feedforward_loop detector"
```

---

## Code Walkthrough

### SparseAutoencoder Class

**Key Methods:**

1. **`__init__`** (line 34-70):
   - Initialize encoder/decoder as simple linear layers
   - Store hyperparameters
   - Xavier weight initialization

2. **`encode`** (line 72-84):
   ```python
   def encode(self, x):
       z = self.encoder(x)  # Linear: input_dim → latent_dim
       z = F.relu(z)        # ReLU: non-negative activations
       return z
   ```
   - Linear transformation followed by ReLU
   - ReLU ensures non-negative latent activations (sparse = many zeros)

3. **`decode`** (line 86-97):
   ```python
   def decode(self, z):
       x_hat = self.decoder(z)  # Linear: latent_dim → input_dim
       return x_hat
   ```
   - Reconstruct original activations from latent space

4. **`forward`** (line 99-111):
   ```python
   def forward(self, x):
       z = self.encode(x)
       x_hat = self.decode(z)
       return x_hat, z
   ```
   - Returns both reconstruction and latent features

5. **`loss`** (line 130-155):
   ```python
   def loss(self, x, x_hat, z):
       recon_loss = F.mse_loss(x_hat, x)

       if self.sparsity_type == "topk":
           sparsity_loss = self._topk_sparsity(z)
       else:
           sparsity_loss = torch.mean(torch.abs(z))

       total_loss = recon_loss + self.sparsity_lambda * sparsity_loss
       return total_loss, recon_loss, sparsity_loss
   ```

### SAETrainer Class

**Key Methods:**

1. **`train_epoch`** (line 186-224):
   - Forward pass through SAE
   - Compute loss
   - Backward pass and optimization
   - Return (total_loss, recon_loss, sparsity_loss)

2. **`validate`** (line 226-259):
   - Same as train_epoch but without parameter updates
   - Used for early stopping

3. **`extract_and_save_latents`** (line 290-324):
   ```python
   # For each batch
   z = self.model.encode(x)  # Get latent features
   # Save to disk
   ```
   - Extracts and saves all latent representations

### InterpretabilityAnalyzer Class

**Key Methods:**

1. **`load_graph_metadata`** (line 57-90):
   - Loads graph paths from disk
   - Extracts motif labels from directory structure
   - Returns: (graph_paths, motif_labels)

2. **`create_node_motif_labels`** (line 92-127):
   ```python
   # For each graph:
   #   Load its activation
   #   Count number of nodes
   #   Assign graph's motif label to all nodes
   # Return: (node_labels, node_graph_ids)
   ```
   - Maps graph-level labels to node-level labels

3. **`compute_pointbiserial_correlations`** (line 129-173):
   ```python
   for motif in motif_types:
       binary_labels = (motif_labels == motif).astype(int)
       for feature in features:
           corr, pval = pointbiserialr(binary_labels, feature_values)
   ```

4. **`compute_mutual_information`** (line 175-224):
   ```python
   for feature in features:
       feature_discrete = np.digitize(feature_values, bins)
       mi = mutual_info_score(motif_labels, feature_discrete)
   ```

5. **`identify_interpretable_features`** (line 226-277):
   ```python
   for feature in features:
       max_corr = max(|correlations with all motifs|)
       target_motif = motif with max_corr

       if max_corr >= 0.5:  # High to one
           other_max = max(|correlations with other motifs|)
           if other_max < 0.2:  # Low to others
               → INTERPRETABLE!
   ```

6. **`plot_motif_feature_heatmap`** (line 279-322):
   - Create motif × feature matrix
   - Select top-50 features by max absolute correlation
   - Plot as seaborn heatmap

### CausalAblationAnalyzer Class

**Purpose**: Test causal effects of features on GNN predictions

**Key Concept**:
```python
# Original forward pass
pred_original = gnn(activation)

# Ablate feature (set to zero)
z = sae.encode(activation)
z[:, feature_idx] = 0  # Remove feature
activation_ablated = sae.decode(z)

# Modified forward pass
pred_ablated = gnn(activation_ablated)

# Effect: how much did the feature matter?
effect = |pred_original - pred_ablated|
```

**Note**: Currently not fully implemented (raises NotImplementedError). Would require:
1. Loading GNN model with custom forward hooks
2. Intercepting layer activations during inference
3. Replacing with ablated versions
4. Measuring prediction changes

---

## Practical Example: Discovering a Feedforward-Loop Feature

### Scenario

We train an SAE on Layer 1 activations and discover:

**Feature 73:**
- Correlation with feedforward_loop: **0.78**
- Correlation with feedback_loop: **-0.05**
- Correlation with single_input_module: **0.02**
- Correlation with cascade: **0.01**

### Analysis

```
Step 1: Identify as Interpretable
  ✓ max_correlation (0.78) >= threshold_high (0.5)
  ✓ max_other_correlation (0.05) < threshold_low (0.2)
  → Feature 73 is interpretable!

Step 2: Understand the Feature
  For nodes in feedforward_loop graphs:
    Mean activation = 0.72
  For nodes in other graphs:
    Mean activation = -0.08

  → Feature 73 strongly activates (positive) for feedforward loops
  → Feature 73 strongly deactivates (negative) for others

Step 3: Visualization
  Plot distribution of Feature 73 across motif types:

  [Histograms would show clear separation between
   feedforward_loop distribution (shifted right)
   and other motifs (shifted left)]

Step 4: Potential Explanation
  Query Layer 1 activations to understand:
    "What patterns in GNN Layer 1 lead to Feature 73 activation?"

  Feature 73 might detect:
    - Specific connectivity patterns in feedforward loops
    - Particular activation correlations between nodes
    - Distinctive message-passing patterns
```

### Interpretation

This is interpretable because:
1. **Specialized**: Activates specifically for one motif type
2. **Consistent**: Shows clear separation from other motifs
3. **Sparse**: Most other features don't need to activate for this
4. **Stable**: Correlation is statistical significant

---

## Common Pitfalls & Solutions

### Pitfall 1: No Interpretable Features Found

**Causes:**
- Threshold too strict (increase to 0.3/0.1)
- SAE not trained well (check reconstruction loss)
- Motifs not well-separated in activation space

**Solutions:**
```python
# Relax thresholds
interpretable_df = analyzer.identify_interpretable_features(
    corr_df,
    threshold_high=0.3,  # Was 0.5
    threshold_low=0.1    # Was 0.2
)

# Check SAE quality
print(f"Recon loss: {test_recon_loss}")  # Should be < 0.1
print(f"Sparsity: {test_sparsity_loss}") # Should be > 0.01
```

### Pitfall 2: High Reconstruction Loss

**Cause**: Sparsity penalty too strong (λ too high)

**Solution**:
```python
# Train SAE with lower sparsity penalty
sae = SparseAutoencoder(..., sparsity_lambda=1e-4)  # Was 1e-3
```

### Pitfall 3: Features Not Sparse Enough

**Cause**: Sparsity penalty too weak (λ too low)

**Solution**:
```python
# Train SAE with higher sparsity penalty
sae = SparseAutoencoder(..., sparsity_lambda=0.01)  # Was 1e-3
```

### Pitfall 4: Node-Motif Label Mismatch

**Cause**: Number of nodes doesn't match SAE latents

**Solution** (in interpretability_analysis.py:499-501):
```python
# Trim to match sizes
min_size = min(len(latents), len(motif_labels))
latents = latents[:min_size]
motif_labels = motif_labels[:min_size]
```

---

## Summary Table

| Component | Input | Process | Output | Purpose |
|-----------|-------|---------|--------|---------|
| GNN | Graphs | Train supervised | Activations | Learn graph representations |
| SAE | Activations | Compress + Sparsity | Latent features | Discover interpretable features |
| Analysis | Latents + Motifs | Correlation/MI | Feature rankings | Identify meaningful patterns |
| Visualization | Rankings | Heatmap/Distribution | PNG plots | Communicate findings |

---

## Next Steps

1. **Run the pipeline**:
   ```bash
   python gnn_train.py
   python sparse_autoencoder.py
   python interpretability_analysis.py
   ```

2. **Analyze results**:
   - Check `outputs/interpretability/interpretable_features_layer*.csv`
   - View `outputs/interpretability/heatmap_layer*.png`
   - Plot distributions for top features

3. **Iterate**:
   - Adjust SAE hyperparameters (sparsity_lambda, latent_dim)
   - Change correlation thresholds
   - Try different sparsity types (L1 vs Top-K)

4. **Extend analysis**:
   - Implement causal ablation experiments
   - Probe SAE decoder: what patterns reconstruct each feature?
   - Combine with attention mechanisms to visualize which nodes activate features
