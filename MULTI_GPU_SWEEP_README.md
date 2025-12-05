# Multi-GPU Hyperparameter Sweep Guide

You have 4 NVIDIA TITAN V GPUs available. Here's how to use them efficiently for hyperparameter sweeps.

## Quick Start

### Option 1: Distributed Multi-GPU (RECOMMENDED)
Best performance - runs independent trials on separate GPUs with true parallel execution:

```bash
python hyperparameter_sweep_distributed.py \
    --num_trials 40 \
    --num_epochs 100 \
    --batch_size 128 \
    --num_gpus 4 \
    --output_dir outputs/sweep_distributed_$(date +%Y%m%d_%H%M%S)
```

### Option 2: Multi-GPU with Joblib
Alternative approach using threading:

```bash
python hyperparameter_sweep_multi_gpu.py \
    --num_trials 40 \
    --num_epochs 100 \
    --batch_size 128 \
    --num_gpus 4 \
    --n_jobs 4 \
    --output_dir outputs/sweep_multi_gpu_$(date +%Y%m%d_%H%M%S)
```

## Performance Comparison

**Single GPU (current notebook):**
- ~40 trials × ~5 min/trial = **~200 minutes total**
- Only 1 GPU utilized

**Multi-GPU Distributed:**
- 40 trials ÷ 4 GPUs = 10 trials per GPU
- 10 trials × ~5 min/trial = **~50 minutes total**
- All 4 GPUs utilized = **4x speedup**

## Why Your Current Training is Slow

Looking at your `nvidia-smi` output:
- GPU 0: 415MiB memory, **0% utilization**
- Problem: Training on GPU but with 0% utilization means:
  - Batch size too small (128 is reasonable, not the issue)
  - Most time spent on CPU operations (data loading, preprocessing)
  - GPU waiting for data

## Improvements in New Scripts

1. **Parallel Execution**: Multiple trials run simultaneously on different GPUs
2. **Better Data Loading**: `num_workers=2` for parallel data loading
3. **Pin Memory**: `pin_memory=True` for faster CPU→GPU transfers
4. **Process Isolation**: Each GPU gets its own Python process (distributed version)

## Monitoring GPU Usage

During training, run this in another terminal:
```bash
watch -n 1 nvidia-smi
```

You should see:
- All 4 GPUs with memory usage (e.g., 400-800 MiB each)
- GPU utilization 30-90% (depends on model size)
- Power usage increase (from 24W idle to 100-200W during training)

## Parameters Explained

- `--num_trials`: Total number of hyperparameter combinations to try
- `--num_epochs`: Maximum epochs per trial (early stopping will stop earlier)
- `--batch_size`: Batch size per trial (128 is good for TITAN V)
- `--num_gpus`: Number of GPUs to use (set to 4 for your machine)
- `--n_jobs`: Number of parallel jobs (only for joblib version)

## Output Files

Both scripts save:
- `best_params.json`: Best hyperparameters found
- `study_info.json`: Summary statistics
- `trials.csv`: All trials with parameters and results
- `all_results.json`: Detailed results with timing info

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python hyperparameter_sweep_distributed.py --batch_size 64 ...
```

### "No graphs found"
Generate graphs first:
```bash
python graph_motif_generator.py
```

### Still seeing 0% GPU utilization
Check:
1. Data loading is bottleneck - increase `num_workers`
2. Model is too small - try larger `hidden_dim` range
3. Use the distributed version (better GPU isolation)

## Example Output

```
==================================================================================
DISTRIBUTED MULTI-GPU GCN HYPERPARAMETER SWEEP
==================================================================================
Number of trials: 40
Max epochs per trial: 100
Batch size: 128
Number of GPUs: 4
==================================================================================

Available GPUs: 4
  GPU 0: NVIDIA TITAN V
  GPU 1: NVIDIA TITAN V
  GPU 2: NVIDIA TITAN V
  GPU 3: NVIDIA TITAN V

Loading data...
Found 4000 graphs
Split: 3200 train, 400 val, 400 test

Starting 4 worker processes...
Worker for GPU 0 started (PID: 12345)
Worker for GPU 1 started (PID: 12346)
Worker for GPU 2 started (PID: 12347)
Worker for GPU 3 started (PID: 12348)

GPU 0: Starting trial 0 with params {'hidden_dim': 64, 'dropout': 0.2, 'learning_rate': 0.001}
GPU 1: Starting trial 1 with params {'hidden_dim': 128, 'dropout': 0.3, 'learning_rate': 0.0005}
GPU 2: Starting trial 2 with params {'hidden_dim': 96, 'dropout': 0.1, 'learning_rate': 0.002}
GPU 3: Starting trial 3 with params {'hidden_dim': 72, 'dropout': 0.25, 'learning_rate': 0.0008}
...
```

## Next Steps

1. **Stop the current notebook cell** (it's running slowly on 1 GPU)
2. **Run the distributed script** using the command above
3. **Monitor with `nvidia-smi`** to verify all GPUs are working
4. **Wait ~50 minutes** for 40 trials (vs 200 minutes on 1 GPU)
5. **Check outputs/** directory for results
