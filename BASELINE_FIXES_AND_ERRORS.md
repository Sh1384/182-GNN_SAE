# MeanMedian Baseline: Error Analysis and Fixes

## The Error You're Experiencing

Based on analysis of the code, the **MeanMedian optimizer error** occurs in one of two scenarios:

### Scenario 1: Device Mismatch (Most Likely)
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices,
cpu and cuda!
```

**Where it happens**: Line 221-223 in `train_epoch()`:
```python
pred = self.model(batch)          # Returns tensor on CPU (bug!)
loss_per_node = self.criterion(pred, batch.y)  # batch.y is on GPU
# ↑ CRASH: pred is CPU, batch.y is GPU
```

**Root cause**: In `MeanMedianBaseline.forward()` [line 73]:
```python
predictions = torch.full((n_nodes,), self.observed_values, dtype=torch.float32)
#           ↑ No device specified = defaults to CPU!
#           But batch.y was moved to GPU by: batch = batch.to(self.device)
```

**Solution**: Specify device when creating tensor:
```python
def forward(self, data: Data) -> torch.Tensor:
    """Predict same value for all masked nodes."""
    n_nodes = data.x.shape[0]
    device = data.x.device  # ← Get device from input
    predictions = torch.full(
        (n_nodes,),
        self.observed_values,
        dtype=torch.float32,
        device=device  # ← ADD THIS!
    )
    return predictions
```

---

### Scenario 2: fit() Device Issues (Less Likely)
```
RuntimeError: CUDA runtime error : device-side assert triggered
```

**Where it happens**: Line 61 in `fit()`:
```python
observed_values = batch.y[observed_mask]
all_observed.extend(observed_values.cpu().numpy())
# If batch.y has gradients (requires_grad=True), .cpu() might fail in some edge cases
```

**Solution**: Add `.detach()` before `.cpu()`:
```python
def fit(self, train_loader: DataLoader):
    """Compute mean or median from training data."""
    all_observed = []

    for batch in train_loader:
        observed_mask = ~batch.mask
        observed_values = batch.y[observed_mask]
        # Add .detach() to ensure no gradient tracking issues
        all_observed.extend(observed_values.detach().cpu().numpy())

    all_observed = np.array(all_observed)

    if self.statistic == 'mean':
        self.observed_values = np.mean(all_observed)
    else:  # median
        self.observed_values = np.median(all_observed)
```

---

## Code Issues Summary

### Issue 1: Device Mismatch in forward()
**File**: `benchmarking.py:71-74`
**Severity**: HIGH (causes crashes)
**Affected line**: 73

#### Current Code (BROKEN):
```python
def forward(self, data: Data) -> torch.Tensor:
    """Predict same value for all masked nodes."""
    n_nodes = data.x.shape[0]
    predictions = torch.full((n_nodes,), self.observed_values, dtype=torch.float32)
    return predictions
```

#### Fixed Code:
```python
def forward(self, data: Data) -> torch.Tensor:
    """Predict same value for all masked nodes."""
    n_nodes = data.x.shape[0]
    # Ensure predictions tensor is on the same device as input data
    device = data.x.device
    predictions = torch.full(
        (n_nodes,),
        self.observed_values,
        dtype=torch.float32,
        device=device
    )
    return predictions
```

**Why this fixes it**:
- `torch.full(..., device=device)` ensures the prediction tensor lives on the same device as the input
- When batch is moved to GPU with `batch.to(self.device)`, both `batch.x` and `batch.y` are on GPU
- The loss computation `criterion(pred, batch.y)` then works because both tensors are on the same device

---

### Issue 2: Potential Gradient Issues in fit()
**File**: `benchmarking.py:52-69`
**Severity**: MEDIUM (rare, edge case)
**Affected line**: 62

#### Current Code (POTENTIALLY BROKEN):
```python
for batch in train_loader:
    observed_mask = ~batch.mask
    observed_values = batch.y[observed_mask]
    all_observed.extend(observed_values.cpu().numpy())
```

#### Fixed Code:
```python
for batch in train_loader:
    observed_mask = ~batch.mask
    observed_values = batch.y[observed_mask]
    # Detach before moving to CPU to avoid gradient tracking issues
    all_observed.extend(observed_values.detach().cpu().numpy())
```

**Why this fixes it**:
- If `batch.y` has `requires_grad=True`, calling `.cpu()` directly can cause issues in some PyTorch versions
- `.detach()` removes the gradient tracking, making it safe to move to CPU
- We don't need gradients here anyway—we're just computing statistics

---

## Data Flow with Fix

```
┌─────────────────────────────────────────────────────┐
│ Batch Creation (BaselineTrainer.train_epoch)        │
├─────────────────────────────────────────────────────┤
│ batch = batch.to(self.device)  # Device = 'cuda'    │
│ batch.x is now on GPU                              │
│ batch.y is now on GPU                              │
│ batch.mask is now on GPU                           │
└────────────┬────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────┐
│ MeanMedian.forward(data)                            │
├─────────────────────────────────────────────────────┤
│ device = data.x.device  # ← Get device from input   │
│ predictions = torch.full(                           │
│     (n_nodes,),                                     │
│     self.observed_values,                           │
│     dtype=torch.float32,                            │
│     device=device  # ← ✓ SAME DEVICE AS INPUT!     │
│ )                                                   │
│ Returns tensor on GPU                              │
└────────────┬────────────────────────────────────────┘
             │
             ↓
┌─────────────────────────────────────────────────────┐
│ Loss Computation                                    │
├─────────────────────────────────────────────────────┤
│ pred = self.model(batch)  # Returns GPU tensor ✓   │
│ loss_per_node = self.criterion(pred, batch.y)      │
│                    # Both on GPU ✓ NO CRASH!       │
│ masked_loss = loss_per_node[batch.mask]  # GPU ✓   │
│ loss = masked_loss.mean()  # Scalar ✓              │
└─────────────────────────────────────────────────────┘
```

---

## Why the Current Code Works (Sometimes)

The current code in `benchmarking.py` handles the optimizer correctly:

```python
class BaselineTrainer:
    def __init__(self, model, device, learning_rate, seed):
        self.model = model.to(device)
        self.device = device
        model_params = list(model.parameters())
        self.optimizer = torch.optim.Adam(model_params, lr=learning_rate) if model_params else None
        # ↑ For MeanMedian: model_params = [] → self.optimizer = None ✓

    def train_epoch(self, train_loader):
        for batch in train_loader:
            if self.optimizer is not None:  # ← Correctly skipped for MeanMedian
                self.optimizer.zero_grad()

            pred = self.model(batch)  # ← Device mismatch happens HERE

            if self.optimizer is not None:  # ← Correctly skipped for MeanMedian
                loss.backward()
                self.optimizer.step()
```

**So the optimizer handling is actually correct!** The issue is purely about device placement of the prediction tensor.

---

## When the Bug Manifests

### If running on CPU:
- ✓ Works fine (everything is on CPU)

### If running on GPU (CUDA):
- ✗ Crashes during loss computation
- Error: `RuntimeError: Expected all tensors to be on the same device`

---

## Complete Fixed Code

### Fix for MeanMedianBaseline.forward() (REQUIRED)

```python
def forward(self, data: Data) -> torch.Tensor:
    """Predict same value for all masked nodes."""
    n_nodes = data.x.shape[0]
    # Ensure predictions tensor is on the same device as input data
    device = data.x.device
    predictions = torch.full(
        (n_nodes,),
        self.observed_values,
        dtype=torch.float32,
        device=device
    )
    return predictions
```

### Fix for MeanMedianBaseline.fit() (OPTIONAL but recommended)

```python
def fit(self, train_loader: DataLoader):
    """
    Compute mean or median from training data.
    """
    all_observed = []

    for batch in train_loader:
        observed_mask = ~batch.mask
        observed_values = batch.y[observed_mask]
        # Detach to avoid any gradient tracking issues
        all_observed.extend(observed_values.detach().cpu().numpy())

    all_observed = np.array(all_observed)

    if self.statistic == 'mean':
        self.observed_values = np.mean(all_observed)
    else:  # median
        self.observed_values = np.median(all_observed)
```

---

## Implementation Instructions

1. **Open**: `benchmarking.py`

2. **Find line 71** (MeanMedianBaseline.forward):
   ```python
   def forward(self, data: Data) -> torch.Tensor:
       """Predict same value for all masked nodes."""
       n_nodes = data.x.shape[0]
   ```

3. **Replace lines 71-74** with the fixed version above

4. **Find line 52** (MeanMedianBaseline.fit):
   ```python
   def fit(self, train_loader: DataLoader):
   ```

5. **Replace line 62** from:
   ```python
   all_observed.extend(observed_values.cpu().numpy())
   ```
   to:
   ```python
   all_observed.extend(observed_values.detach().cpu().numpy())
   ```

6. **Save and test**:
   ```bash
   python benchmarking.py --seeds 2 --epochs 20
   ```

---

## Testing Checklist

After applying fixes:
- [ ] MeanMedian baseline runs without device errors
- [ ] Optimizer is correctly None: `trainer.optimizer is None`
- [ ] Forward pass returns tensor on correct device
- [ ] Loss computation works without device mismatch errors
- [ ] Validation and test phases complete successfully

---

## Summary Table

| Issue | Location | Severity | Fix | Required |
|-------|----------|----------|-----|----------|
| Device mismatch in forward() | Line 73 | HIGH | Add `device=data.x.device` to `torch.full()` | **YES** |
| Gradient tracking in fit() | Line 62 | MEDIUM | Add `.detach()` before `.cpu()` | Recommended |

