"""
Distributed Multi-GPU Hyperparameter Sweep for GCN Training

Uses torch.multiprocessing to run independent trials on separate GPUs simultaneously.
This provides true parallel execution with isolated GPU contexts.
"""

import argparse
import json
import os
import time
import queue
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler

# Import from gnn_train_copy
from gnn_train_copy import (
    GraphDataset,
    GCNModel,
    GNNTrainer,
    collate_fn,
    load_all_graphs,
    split_data,
)


def worker_process(
    gpu_id: int,
    trial_queue: mp.Queue,
    result_queue: mp.Queue,
    train_paths: List,
    val_paths: List,
    config: Dict,
):
    """
    Worker process that runs trials on a specific GPU.

    Args:
        gpu_id: GPU ID for this worker
        trial_queue: Queue of trials to process
        result_queue: Queue to put results
        train_paths: Training graph paths
        val_paths: Validation graph paths
        config: Configuration dictionary
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    print(f"Worker for GPU {gpu_id} started (PID: {os.getpid()})")

    while True:
        try:
            # Get trial from queue (with timeout to check for completion)
            trial_data = trial_queue.get(timeout=1)

            if trial_data is None:  # Poison pill
                break

            trial_number = trial_data['trial_number']
            params = trial_data['params']

            print(f"GPU {gpu_id}: Starting trial {trial_number} with params {params}")

            # Create datasets
            train_dataset = GraphDataset(
                train_paths,
                mask_prob=config['mask_prob'],
                seed=config['seed']
            )
            val_dataset = GraphDataset(
                val_paths,
                mask_prob=config['mask_prob'],
                seed=config['seed'] + 1
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=2,
                pin_memory=True
            )

            # Create model and trainer
            model = GCNModel(
                input_dim=2,
                hidden_dim=params['hidden_dim'],
                output_dim=1,
                dropout=params['dropout']
            )
            trainer = GNNTrainer(
                model,
                device=device,
                learning_rate=params['learning_rate'],
                seed=config['seed']
            )

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            early_stopping_patience = 25

            start_time = time.time()

            for epoch in range(config['num_epochs']):
                train_loss = trainer.train_epoch(train_loader)
                val_loss = trainer.validate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        break

            elapsed_time = time.time() - start_time

            # Put result in queue
            result_queue.put({
                'trial_number': trial_number,
                'gpu_id': gpu_id,
                'value': best_val_loss,
                'params': params,
                'status': 'COMPLETE',
                'elapsed_time': elapsed_time
            })

            print(f"GPU {gpu_id}: Completed trial {trial_number} "
                  f"(val_loss={best_val_loss:.6f}, time={elapsed_time:.1f}s)")

        except queue.Empty:
            continue
        except Exception as e:
            result_queue.put({
                'trial_number': trial_number if 'trial_number' in locals() else -1,
                'gpu_id': gpu_id,
                'status': 'FAILED',
                'error': str(e)
            })
            error_msg = str(e)
            print(f"GPU {gpu_id}: Trial failed - {error_msg}")
            import traceback
            traceback.print_exc()

    print(f"Worker for GPU {gpu_id} shutting down")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Distributed Multi-GPU Hyperparameter Sweep for GCN'
    )
    parser.add_argument('--num_trials', type=int, default=20,
                        help='Number of trials')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Max epochs per trial')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--mask_prob', type=float, default=0.2,
                        help='Masking probability')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_gpus', type=int, default=4,
                        help='Number of GPUs to use')
    parser.add_argument('--output_dir', type=str,
                        default='outputs/hyperparameter_sweep_distributed',
                        help='Output directory')
    args = parser.parse_args()

    # Enable multiprocessing
    mp.set_start_method('spawn', force=True)

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DISTRIBUTED MULTI-GPU GCN HYPERPARAMETER SWEEP")
    print("=" * 80)
    print(f"Number of trials: {args.num_trials}")
    print(f"Max epochs per trial: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of GPUs: {args.num_gpus}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No CUDA devices available!")
        return

    available_gpus = torch.cuda.device_count()
    print(f"\nAvailable GPUs: {available_gpus}")

    if available_gpus < args.num_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} available")
        args.num_gpus = available_gpus

    for i in range(args.num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load and split data
    print("\nLoading data...")
    all_graph_paths = load_all_graphs(single_motif_only=True)
    print(f"Found {len(all_graph_paths)} graphs")

    if len(all_graph_paths) == 0:
        print("Error: No graphs found. Please run graph_motif_generator.py first.")
        return

    train_paths, val_paths, test_paths = split_data(
        all_graph_paths,
        seed=args.seed,
        stratify_by_motif=True,
        equal_counts_per_motif=True
    )
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

    # Create Optuna study
    print("\nCreating Optuna study...")
    sampler = TPESampler(seed=args.seed, n_startup_trials=5)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        study_name='gcn_distributed_optimization'
    )

    # Generate all trials upfront
    print("Generating trial configurations...")
    trials = []
    for i in range(args.num_trials):
        trial = study.ask()

        # Manually suggest parameters since ask() doesn't auto-suggest
        params = {
            'hidden_dim': trial.suggest_int('hidden_dim', 16, 256, step=8),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
        }

        trials.append({
            'trial_number': i,
            'params': params,
            'trial_obj': trial
        })

    # Create queues
    trial_queue = mp.Queue()
    result_queue = mp.Queue()

    # Populate trial queue (only send necessary data, not trial objects)
    for trial_data in trials:
        trial_queue.put({
            'trial_number': trial_data['trial_number'],
            'params': trial_data['params']
        })

    # Add poison pills for workers
    for _ in range(args.num_gpus):
        trial_queue.put(None)

    # Configuration
    config = {
        'batch_size': args.batch_size,
        'mask_prob': args.mask_prob,
        'num_epochs': args.num_epochs,
        'seed': args.seed,
    }

    # Start worker processes
    print(f"\nStarting {args.num_gpus} worker processes...")
    processes = []
    for gpu_id in range(args.num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, trial_queue, result_queue, train_paths, val_paths, config)
        )
        p.start()
        processes.append(p)

    # Collect results
    print("\nCollecting results...")
    results = []
    for i in range(args.num_trials):
        result = result_queue.get()
        results.append(result)

        # Update study with trial object
        if result['status'] == 'COMPLETE':
            trial_obj = trials[result['trial_number']]['trial_obj']
            study.tell(trial_obj, result['value'])

        # Print progress
        completed = len([r for r in results if r['status'] == 'COMPLETE'])
        failed = len([r for r in results if r['status'] == 'FAILED'])
        print(f"Progress: {len(results)}/{args.num_trials} trials "
              f"({completed} completed, {failed} failed)")

    # Wait for all workers to finish
    for p in processes:
        p.join()

    # Print final results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)

    completed_trials = [r for r in results if r['status'] == 'COMPLETE']
    failed_trials = [r for r in results if r['status'] == 'FAILED']

    print(f"\nCompleted trials: {len(completed_trials)}")
    print(f"Failed trials: {len(failed_trials)}")

    if completed_trials:
        best_trial = min(completed_trials, key=lambda x: x['value'])

        print(f"\nBest Trial: {best_trial['trial_number']}")
        print(f"Best Validation Loss: {best_trial['value']:.6f}")
        print(f"GPU used: {best_trial['gpu_id']}")
        print(f"Time: {best_trial['elapsed_time']:.1f}s")
        print("\nBest Hyperparameters:")
        for key, value in best_trial['params'].items():
            print(f"  {key}: {value}")

        # Calculate statistics
        total_time = sum(r['elapsed_time'] for r in completed_trials)
        avg_time = total_time / len(completed_trials)
        print(f"\nTotal training time: {total_time:.1f}s")
        print(f"Average time per trial: {avg_time:.1f}s")

        # Save results
        print(f"\nSaving results to {args.output_dir}...")

        with open(output_path / "best_params.json", 'w') as f:
            json.dump(best_trial['params'], f, indent=2)

        with open(output_path / "study_info.json", 'w') as f:
            json.dump({
                'best_value': best_trial['value'],
                'best_trial': best_trial['trial_number'],
                'best_gpu': best_trial['gpu_id'],
                'n_trials': args.num_trials,
                'n_complete_trials': len(completed_trials),
                'n_failed_trials': len(failed_trials),
                'total_time': total_time,
                'avg_time_per_trial': avg_time
            }, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_df.to_csv(output_path / "trials.csv", index=False)

        # Save all results with timing info
        with open(output_path / "all_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✓ Saved results to {output_path}")
    else:
        print("\nNo trials completed successfully!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
