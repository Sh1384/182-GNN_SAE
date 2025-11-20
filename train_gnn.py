import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn_train import (
    GraphDataset, GCNModel, GATModel, GNNTrainer,
    load_all_graphs, split_data, collate_fn
)


def train_with_metrics(num_epochs: int = 20, device: str = 'cuda', 
                      batch_size: int = 32, learning_rate: float = 1e-3,
                      mask_prob: float = 0.3, seed: int = 42,
                      model_type: str = "GCN"):
    
    print(f"Using device: {device}")
    print(f"Loading single-motif graphs from ./virtual_graphs/data/")
    
    all_graph_paths = load_all_graphs(single_motif_only=True)
    print(f"Found {len(all_graph_paths)} graphs")
    
    if len(all_graph_paths) == 0:
        print("Error: No graphs found. Please run graph_motif_generator.py first.")
        return
    
    train_paths, val_paths, test_paths = split_data(all_graph_paths, seed=seed)
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    train_dataset = GraphDataset(train_paths, mask_prob=mask_prob, seed=seed)
    val_dataset = GraphDataset(val_paths, mask_prob=mask_prob, seed=seed + 1)
    test_dataset = GraphDataset(test_paths, mask_prob=mask_prob, seed=seed + 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    
    model_type_upper = model_type.upper()
    if model_type_upper == "GAT":
        model = GATModel(
            input_dim=2,
            hidden_dim=16,
            output_dim=1,
            dropout=0.2,
            num_heads=4,
            edge_dim=1
        )
        model_name = "gat_model.pt"
        print("\nInitialized GAT model (multi-head attention)")
    else:
        model = GCNModel(input_dim=2, hidden_dim=64, output_dim=1, dropout=0.2)
        model_name = "gnn_model.pt"
        model_type_upper = "GCN"
        print("\nInitialized GCN model")

    trainer = GNNTrainer(model, device=device, learning_rate=learning_rate, seed=seed)
    
    metrics = {
        'model_type': model_type_upper,
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'train_mse': [],
        'val_mse': [],
        'test_mse': [],
        'train_mae': [],
        'val_mae': [],
        'test_mae': []
    }
    
    def compute_metrics(loader, split_name):
        trainer.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0
        num_nodes = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = trainer.model(batch)
                
                loss_per_node = trainer.criterion(pred, batch.y)
                masked_loss = loss_per_node[batch.mask]
                
                if len(masked_loss) > 0:
                    loss = masked_loss.mean()
                    mse = loss_per_node[batch.mask].mean()
                    mae = torch.abs(pred[batch.mask] - batch.y[batch.mask]).mean()
                    
                    total_loss += loss.item()
                    total_mse += mse.item()
                    total_mae += mae.item()
                    num_batches += 1
                    num_nodes += batch.mask.sum().item()
        
        return {
            'loss': total_loss / max(num_batches, 1),
            'mse': total_mse / max(num_batches, 1),
            'mae': total_mae / max(num_batches, 1)
        }
    
    print(f"\nTraining {model_type_upper} model...")
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss = trainer.train_epoch(train_loader)
        
        train_metrics = compute_metrics(train_loader, 'train')
        val_metrics = compute_metrics(val_loader, 'val')
        test_metrics = compute_metrics(test_loader, 'test')
        
        metrics['train_loss'].append(train_metrics['loss'])
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['test_loss'].append(test_metrics['loss'])
        metrics['train_mse'].append(train_metrics['mse'])
        metrics['val_mse'].append(val_metrics['mse'])
        metrics['test_mse'].append(test_metrics['mse'])
        metrics['train_mae'].append(train_metrics['mae'])
        metrics['val_mae'].append(val_metrics['mae'])
        metrics['test_mae'].append(test_metrics['mae'])
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}")
            print(f"  Test  - Loss: {test_metrics['loss']:.4f}, MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            trainer.save_model(f"checkpoints/{model_name}")
    
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/training_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    motif_metrics = compute_motif_metrics(trainer.model, device, all_graph_paths, seed)
    with open("outputs/motif_metrics.json", 'w') as f:
        json.dump(motif_metrics, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Metrics saved to outputs/training_metrics.json")
    print(f"Motif-specific metrics saved to outputs/motif_metrics.json")
    
    return metrics, motif_metrics


def compute_motif_metrics(model, device, all_graph_paths, seed):
    from gnn_train import GraphDataset
    from torch.utils.data import DataLoader
    
    motif_types = ['feedforward_loop', 'feedback_loop', 'single_input_module', 'cascade']
    motif_metrics = {motif: {'mse': [], 'mae': []} for motif in motif_types}
    
    model.eval()
    criterion = torch.nn.MSELoss(reduction='none')
    
    for motif_type in motif_types:
        motif_paths = [p for p in all_graph_paths if motif_type in str(p)]
        if len(motif_paths) == 0:
            continue
        
        motif_dataset = GraphDataset(motif_paths, mask_prob=0.3, seed=seed)
        motif_loader = DataLoader(motif_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for batch in motif_loader:
                batch = batch.to(device)
                pred = model(batch)
                
                loss_per_node = criterion(pred, batch.y)
                masked_loss = loss_per_node[batch.mask]
                
                if len(masked_loss) > 0:
                    mse = masked_loss.mean().item()
                    mae = torch.abs(pred[batch.mask] - batch.y[batch.mask]).mean().item()
                    
                    motif_metrics[motif_type]['mse'].append(mse)
                    motif_metrics[motif_type]['mae'].append(mae)
    
    for motif in motif_metrics:
        if len(motif_metrics[motif]['mse']) > 0:
            motif_metrics[motif]['mean_mse'] = float(np.mean(motif_metrics[motif]['mse']))
            motif_metrics[motif]['mean_mae'] = float(np.mean(motif_metrics[motif]['mae']))
            motif_metrics[motif]['std_mse'] = float(np.std(motif_metrics[motif]['mse']))
            motif_metrics[motif]['std_mae'] = float(np.std(motif_metrics[motif]['mae']))
    
    return motif_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN (GCN or GAT) with metric logging.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="GCN",
        choices=["GCN", "GAT", "gcn", "gat"],
        help="Model architecture to train."
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Warning: CUDA not available, using CPU")
    
    metrics, motif_metrics = train_with_metrics(
        num_epochs=args.epochs,
        device=device,
        batch_size=32,
        learning_rate=1e-3,
        mask_prob=0.3,
        seed=42,
        model_type=args.model_type
    )

