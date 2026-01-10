# Comprehensive Baseline Methods Analysis

## Executive Summary
This document provides in-depth analysis of the two baseline methods (MLP and MeanMedian) in `benchmarking.py`, including:
1. How data flows through each model
2. Why the optimizer error occurs with MeanMedian
3. Root causes and solutions

---

## 1. MLP BASELINE (Multi-Layer Perceptron)

### Architecture Overview
```
Input Layer: 2 features
    ├─ masked_expression (continuous: 0-1)
    └─ mask_flag (binary: 1 if observed, 0 if masked)
            ↓
Hidden Layer 1: 128 units
    ├─ Linear(2 → 128)
    ├─ ReLU activation
    └─ Dropout(p=0.2)
            ↓
Hidden Layer 2: 64 units
    ├─ Linear(128 → 64)
    ├─ ReLU activation
    └─ Dropout(p=0.2)
            ↓
Output Layer: 1 unit
    ├─ Linear(64 → 1)
    └─ Squeeze to scalar prediction
            ↓
Output: Single predicted value per node
```

### Data Flow Through MLP

#### Step 1: Input Preparation (in benchmarking.py:323-325)
```python
train_dataset = GraphDataset(train_paths, mask_prob=0.2, seed=seed)
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn  # Batches multiple graphs into single Data object
)
```

#### Step 2: Batch Composition
Each batch contains:
```python
batch.x          # Shape: [total_nodes_in_batch, 2]
                 # Column 0: masked expression (0 if masked, normalized otherwise)
                 # Column 1: mask_flag (1.0 if observed, 0.0 if masked)

batch.y          # Shape: [total_nodes_in_batch]
                 # Ground truth expression values for all nodes

batch.mask       # Shape: [total_nodes_in_batch, bool]
                 # True = node is masked (to be predicted)
                 # False = node is observed (used for training)

batch.edge_index # Shape: [2, num_edges] - NOT used by MLP
batch.batch      # Graph indices for each node
```

#### Step 3: Forward Pass (MLPBaseline.forward(), lines 91-102)
```python
def forward(self, data: Data) -> torch.Tensor:
    x = data.x  # [batch_nodes, 2]

    # First hidden layer
    x = F.relu(self.fc1(x))      # [batch_nodes, 128]
    x = self.dropout(x)           # [batch_nodes, 128]

    # Second hidden layer
    x = F.relu(self.fc2(x))      # [batch_nodes, 64]
    x = self.dropout(x)           # [batch_nodes, 64]

    # Output layer
    x = self.fc3(x)              # [batch_nodes, 1]
    return x.squeeze(-1)         # [batch_nodes] - removes last dimension
```

#### Step 4: Loss Computation (BaselineTrainer.train_epoch(), lines 221-223)
```python
pred = self.model(batch)  # [batch_nodes]
loss_per_node = self.criterion(pred, batch.y)  # MSE per node
masked_loss = loss_per_node[batch.mask]  # Only loss on MASKED nodes
loss = masked_loss.mean()  # Single scalar loss
```

### Why MLP Works as a Baseline

| Aspect | GCN/GAT | MLP |
|--------|---------|-----|
| Uses edges | ✓ Yes | ✗ No |
| Uses topology | ✓ Yes | ✗ No |
| Uses only features | ✓ Yes | ✓ Yes |
| Has trainable params | ✓ Yes | ✓ Yes |
| Can learn from masking pattern | ✓ Yes | ✓ Yes |

**Key insight**: If MLP performs similarly to GCN/GAT, it suggests the graph structure doesn't help—contradicting the research hypothesis.

---

## 2. MEANMEDIAN BASELINE (Non-Parametric)

### Architecture
This is a **stateless, non-trainable** model that predicts a single constant value for all nodes.

```
Training (fit phase):
    ├─ Load all training data
    ├─ Identify observed nodes (batch.mask == False)
    ├─ Extract their expression values (batch.y)
    ├─ Compute mean (or median) across ALL observed values
    └─ Store as single constant: self.observed_values

Prediction (forward phase):
    ├─ Receive batch of nodes
    ├─ Return same constant for ALL nodes
    └─ No computation, no gradients
```

### Detailed Implementation

#### Fit Phase (lines 52-68)
```python
def fit(self, train_loader: DataLoader):
    """Compute mean or median from training data."""
    all_observed = []

    for batch in train_loader:
        # batch.mask is True for masked nodes, False for observed
        observed_mask = ~batch.mask  # Invert: True for observed
        observed_values = batch.y[observed_mask]  # Extract observed values
        all_observed.extend(observed_values.cpu().numpy())

    all_observed = np.array(all_observed)

    if self.statistic == 'mean':
        self.observed_values = np.mean(all_observed)  # Single float
    else:
        self.observed_values = np.median(all_observed)  # Single float
```

**Example**:
- Training data has 3200 observed nodes across all training graphs
- Their values: [0.1, 0.3, 0.2, 0.5, ...]
- Mean computed: 0.35 (example)
- `self.observed_values = 0.35`

#### Prediction Phase (lines 70-74)
```python
def forward(self, data: Data) -> torch.Tensor:
    """Predict same value for all masked nodes."""
    n_nodes = data.x.shape[0]
    predictions = torch.full(
        (n_nodes,),                    # Shape: [n_nodes]
        self.observed_values,          # Value: 0.35 (from fit)
        dtype=torch.float32
    )
    return predictions
```

**Example**:
- Batch has 50 nodes
- Returns tensor: [0.35, 0.35, 0.35, ..., 0.35] (50 times)

### Why MeanMedian is "No Parameters"

```python
class MeanMedianBaseline(nn.Module):
    def __init__(self, statistic: str = 'mean'):
        super().__init__()
        self.statistic = statistic
        self.observed_values = None  # ← This is NOT a nn.Parameter!
```

When checking for parameters:
```python
list(model.parameters())  # Returns [] (empty list!)
```

This is intentional—MeanMedian should NOT have trainable parameters. It's a **data-dependent baseline**, not a learned model.

---

## 3. THE OPTIMIZER ISSUE

### Root Cause: Why the Check Exists

In `BaselineTrainer.__init__` (lines 198-204):
```python
def __init__(self, model: nn.Module, device: str = 'cuda',
             learning_rate: float = 1e-3, seed: int = 42):
    self.model = model.to(device)
    self.device = device

    # Only create optimizer if model has parameters
    model_params = list(model.parameters())
    self.optimizer = torch.optim.Adam(model_params, lr=learning_rate) if model_params else None
    self.criterion = nn.MSELoss(reduction='none')
```

**Why this check is needed**:
- MeanMedian has no parameters → cannot create optimizer
- Would cause: `RuntimeError: optimizer got an empty parameter list`
- Solution: Set `self.optimizer = None` and skip optimization steps

### How the Code Handles It

In `BaselineTrainer.train_epoch` (lines 216-227):
```python
if self.optimizer is not None:  # ← Guards against None optimizer
    self.optimizer.zero_grad()

pred = self.model(batch)
loss = criterion(pred, batch.y)

if self.optimizer is not None:  # ← Skips backward/step for MeanMedian
    loss.backward()
    self.optimizer.step()
```

This is **correct** for MeanMedian—no gradients needed!

### Potential Error Sources

However, you might encounter errors in these scenarios:

#### Issue 1: Device Mismatch in fit()
```python
# In MeanMedianBaseline.fit(), line 61:
observed_values = batch.y[observed_mask]
all_observed.extend(observed_values.cpu().numpy())  # ← Works fine

# But if batch is never moved to device:
batch = batch.to(device)  # NOT done in fit()!
# This could cause issues if batch.y is on GPU
```

**Fix**: Ensure batch is on CPU before calling `.numpy()`:
```python
observed_values = batch.y[observed_mask].cpu().numpy()
```

#### Issue 2: Device Mismatch in forward()
```python
# In MeanMedianBaseline.forward(), line 73:
predictions = torch.full((n_nodes,), self.observed_values, dtype=torch.float32)
# This tensor is created on CPU by default!

# But batch.y is on GPU:
loss = criterion(pred, batch.y)  # GPU tensor
# RuntimeError: Expected all tensors to be on the same device
```

**Fix**: Move predictions to correct device:
```python
def forward(self, data: Data) -> torch.Tensor:
    n_nodes = data.x.shape[0]
    predictions = torch.full(
        (n_nodes,),
        self.observed_values,
        dtype=torch.float32,
        device=data.x.device  # ← Use same device as input
    )
    return predictions
```

#### Issue 3: Calling .to(device) on MeanMedian
```python
# In BaselineTrainer.__init__, line 200:
self.model = model.to(device)

# This moves nn.Module to device, but stored numpy value stays on CPU
# When forward() is called, the tensor is created on CPU
```

---

## 4. COMPARISON TABLE: MLP vs MeanMedian

| Property | MLP | MeanMedian |
|----------|-----|-----------|
| **Trainable Parameters** | Yes (384 params) | No (0 params) |
| **Uses Node Features** | Yes | No |
| **Uses Graph Topology** | No | No |
| **Learns Masking Pattern** | Yes | No |
| **Requires fit() phase** | No | Yes |
| **Has Optimizer** | Yes | No |
| **Training Time** | ~20-50 epochs | 0 (no training) |
| **Computation per forward** | ~O(features) | O(1) |
| **Expected Performance** | Better than MeanMedian | Worse (simple baseline) |

---

## 5. EXPECTED PERFORMANCE HIERARCHY

```
Best Performance
        ↑
     GCN/GAT (exploit graph structure)
        ↑
      MLP (learns from features + masking pattern)
        ↑
  MeanMedian (predicts constant value)
        ↓
Worst Performance (no learning)
```

---

## 6. DATA FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA PREPARATION                    │
├─────────────────────────────────────────────────────────────┤
│  GraphDataset loads 10-node graphs with expression values    │
│  Masks 20% of nodes randomly                                 │
│  Creates features: [masked_expr, mask_flag]                 │
└────────────┬────────────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────────────┐
│                    BATCH CREATION                            │
├─────────────────────────────────────────────────────────────┤
│  Batches: Collate_fn merges multiple graphs                 │
│  Shape: [128 nodes × 128 batch_size on average]             │
│  batch.x: [N, 2] → node features                            │
│  batch.y: [N] → ground truth                                │
│  batch.mask: [N, bool] → masking indicator                  │
│  batch.edge_index: [2, E] → graph edges (ignored by MLP)    │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────────────┬────────────────────────┐
             ↓                     ↓                        ↓
        ┌────────┐          ┌─────────┐            ┌──────────────┐
        │   GCN  │          │   MLP   │            │ MeanMedian   │
        ├────────┤          ├─────────┤            ├──────────────┤
        │ GCN    │          │Linear 2→128          │ fit(train):  │
        │ Layer1 │          │+ ReLU + Dropout      │ mean(y_obs)  │
        │        │          │        ↓             │              │
        │ GCN    │          │Linear 128→64         │ forward():   │
        │ Layer2 │          │+ ReLU + Dropout      │ repeat(value)│
        │        │          │        ↓             │              │
        │ GCN    │          │Linear 64→1           │ No optimizer │
        │ Layer3 │          │        ↓             │              │
        │ OUTPUT │          │squeeze() → [N]       │ No gradients │
        └─┬──────┘          └────────┬─────────────┘              │
          │                         │                             │
          └─────────────┬───────────┴─────────────────────────────┘
                        │
                        ↓
            ┌───────────────────────────────┐
            │   Loss Computation             │
            ├───────────────────────────────┤
            │ loss_per_node = MSE(pred, y)  │
            │ masked_loss = loss[mask==True]│
            │ final_loss = mean(masked_loss)│
            └───────────┬───────────────────┘
                        │
                        ↓
            ┌───────────────────────────────┐
            │   Optimizer Step (if exists)   │
            ├───────────────────────────────┤
            │ loss.backward()                │
            │ optimizer.step()               │
            └───────────────────────────────┘
```

---

## 7. FIXING THE DEVICE ISSUE

### Recommended Changes

#### Fix 1: Update MeanMedian.forward()
```python
def forward(self, data: Data) -> torch.Tensor:
    """Predict same value for all masked nodes."""
    n_nodes = data.x.shape[0]
    device = data.x.device  # Get device from input
    predictions = torch.full(
        (n_nodes,),
        self.observed_values,
        dtype=torch.float32,
        device=device  # ← ADD THIS LINE
    )
    return predictions
```

#### Fix 2: Ensure fit() handles device properly
```python
def fit(self, train_loader: DataLoader):
    """Compute mean or median from training data."""
    all_observed = []

    for batch in train_loader:
        observed_mask = ~batch.mask
        observed_values = batch.y[observed_mask]
        all_observed.extend(observed_values.detach().cpu().numpy())  # ← Add .detach()

    all_observed = np.array(all_observed)

    if self.statistic == 'mean':
        self.observed_values = np.mean(all_observed)
    else:
        self.observed_values = np.median(all_observed)
```

---

## 8. TESTING THE BASELINES

### Test MLP
```python
# Should train normally with optimizer updates
model = MLPBaseline(input_dim=2)
trainer = BaselineTrainer(model, device='cuda')
# Should not encounter optimizer errors
```

### Test MeanMedian
```python
# Should fit without optimizer
model = MeanMedianBaseline(statistic='mean')
model.fit(train_loader)
trainer = BaselineTrainer(model, device='cuda')
# optimizer should be None
assert trainer.optimizer is None
# Should handle forward pass correctly
```

---

## Summary of Findings

1. **MLP**: Standard PyTorch module with trainable parameters. Data flows correctly through 3 layers.
2. **MeanMedian**: Non-parametric baseline that stores single computed value.
3. **Data Flow**: Node features only (no graph structure) → output prediction per node.
4. **Optimizer Issue**: Correctly handled by checking if parameters exist.
5. **Potential Bugs**: Device mismatch in MeanMedian forward() and fit() phases.
6. **Fixes**: Add `device=data.x.device` to torch.full() call.
