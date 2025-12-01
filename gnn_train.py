"""
GNN Training Module for Node Value Prediction

Trains a two-layer Graph Convolutional Network (GCN) to predict missing node values
from partially observed synthetic graphs. Saves trained models and layer activations
for downstream interpretability analysis.
"""

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from tqdm import tqdm


MOTIF_LABELS = [
    "cascade",
    "feedback_loop",
    "feedforward_loop",
    "single_input_module",
    "mixed_motif",
    "unknown"
]
MOTIF_TO_ID = {label: idx for idx, label in enumerate(MOTIF_LABELS)}


def _infer_motif_label(graph_path: Path) -> str:
    """Infer motif label from a path using directory structure."""
    parts = graph_path.parts
    if "single_motif_graphs" in parts:
        idx = parts.index("single_motif_graphs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if "mixed_motif_graphs" in parts:
        return "mixed_motif"
    return "unknown"


class GraphDataset(Dataset):
    def __init__(self, graph_paths: List[Path], mask_prob: float = 0.3, seed: int = 42):
        self.graph_paths = graph_paths
        self.mask_prob = mask_prob
        self.base_seed = seed

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> Data:
        graph_path = self.graph_paths[idx]
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        graph_id = int(graph_path.stem.split('_')[1])
        motif_label = _infer_motif_label(graph_path)
        motif_id = MOTIF_TO_ID.get(motif_label, MOTIF_TO_ID["unknown"])

        local_seed = self.base_seed + graph_id
        local_rng = np.random.default_rng(local_seed)

        import networkx as nx
        n_nodes = len(G.nodes())
        W = nx.to_numpy_array(G, weight='weight')

        expression = self._simulate_expression(W, rng=local_rng)

        mask = local_rng.random(n_nodes) < self.mask_prob
        mask = torch.tensor(mask, dtype=torch.bool)

        expression_tensor = torch.tensor(expression, dtype=torch.float32)
        masked_expression = expression_tensor.clone()
        masked_expression[mask] = 0.0

        if masked_expression.max() > 0:
            masked_expression = masked_expression / (masked_expression.max() + 1e-8)

        mask_flag = (~mask).float().unsqueeze(1)
        x = torch.cat([masked_expression.unsqueeze(1), mask_flag], dim=1)

        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=expression_tensor,
            mask=mask,
            num_nodes=n_nodes
        )
        data.motif_id = torch.tensor([motif_id], dtype=torch.long)
        data.graph_id = torch.tensor([graph_id], dtype=torch.long)

        return data

    def _simulate_expression(self, W: np.ndarray, rng: np.random.Generator, 
                             steps: int = 50, gamma: float = 0.3, 
                             noise_std: float = 0.01) -> np.ndarray:
        n_nodes = W.shape[0]
        x = rng.uniform(0, 1, size=n_nodes)

        for _ in range(steps):
            weighted_input = W @ x
            sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
            noise = rng.normal(0, noise_std, size=n_nodes)
            x = (1 - gamma) * x + gamma * sigmoid_input + noise
            x = np.clip(x, 0, 1)

        return x


class GCNModel(nn.Module):
    """
    Three-layer Graph Convolutional Network for node value prediction.

    Architecture:
        - Layer 1: GCNConv(2 -> 128) + ReLU + Dropout (1-hop neighborhoods)
        - Layer 2: GCNConv(128 -> 64) + ReLU + Dropout (2-hop neighborhoods)
        - Layer 3: GCNConv(64 -> 1) (3-hop neighborhoods, final prediction)
    """

    def __init__(self, input_dim: int = 2, hidden_dim1: int = 128,
                 hidden_dim2: int = 64, output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the GCN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim1: First hidden layer dimension
            hidden_dim2: Second hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(GCNModel, self).__init__()

        # Disable internal normalization to support signed edge weights
        self.conv1 = GCNConv(input_dim, hidden_dim1, normalize=False)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, normalize=False)
        self.conv3 = GCNConv(hidden_dim2, output_dim, normalize=False)
        self.dropout = dropout

        # Storage for activations
        self.layer1_activations = None
        self.layer2_activations = None
        self.layer3_activations = None

    def forward(self, data: Data, store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass through the 3-layer GCN.

        Args:
            data: PyG Data object
            store_activations: Whether to store layer activations

        Returns:
            Predicted node values
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)

        # Layer 1: GCNConv + ReLU + Dropout (1-hop neighborhoods)
        h1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        h1 = F.relu(h1)

        if store_activations:
            self.layer1_activations = h1.detach()

        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2: GCNConv + ReLU + Dropout (2-hop neighborhoods)
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)
        h2 = F.relu(h2)

        if store_activations:
            self.layer2_activations = h2.detach()

        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Layer 3: GCNConv (3-hop neighborhoods, final prediction)
        h3 = self.conv3(h2, edge_index, edge_weight=edge_weight)

        if store_activations:
            self.layer3_activations = h3.detach()

        return h3.squeeze(-1)

    def get_activations(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get stored layer activations.

        Returns:
            Tuple of (layer1_activations, layer2_activations, layer3_activations)
        """
        return self.layer1_activations, self.layer2_activations, self.layer3_activations


class GNNTrainer:
    """
    Trainer class for GCN model on synthetic graphs.

    Handles training loop, validation, and saving of models and activations.
    """

    def __init__(self, model: GCNModel, device: str = 'cuda',
                 learning_rate: float = 1e-3, seed: int = 42):
        """
        Initialize the trainer.

        Args:
            model: GCN model to train
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            seed: Random seed for reproducibility
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='none')

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batch)

            # Compute loss only on masked nodes
            loss_per_node = self.criterion(pred, batch.y)
            masked_loss = loss_per_node[batch.mask]

            if len(masked_loss) > 0:
                loss = masked_loss.mean()

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                pred = self.model(batch)

                # Compute loss only on masked nodes
                loss_per_node = self.criterion(pred, batch.y)
                masked_loss = loss_per_node[batch.mask]

                if len(masked_loss) > 0:
                    loss = masked_loss.mean()
                    total_loss += loss.item()
                    num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_model(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def extract_and_save_activations(self, data_loader: DataLoader,
                                     output_dir: str, split_name: str):
        """
        Extract and save layer2 (64-dim) activations for all graphs.

        Args:
            data_loader: DataLoader for graphs
            output_dir: Base output directory
            split_name: Name of data split (train/val/test)
        """
        self.model.eval()

        # Only create layer2 directory (64-dimensional activations)
        layer2_dir = Path(output_dir) / "activations" / "layer2" / split_name
        layer2_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {split_name} activations"):
                batch = batch.to(self.device)

                # Forward pass with activation storage
                _ = self.model(batch, store_activations=True)
                # Only retrieve layer2 activations (64-dim)
                h2 = self.model.layer2_activations

                # Split batch back into individual graphs
                if hasattr(batch, 'batch'):
                    # Handle batched data
                    batch_indices = batch.batch
                    unique_batches = torch.unique(batch_indices)

                    for b in unique_batches:
                        mask = batch_indices == b

                        # Extract layer2 activations for this graph
                        h2_graph = h2[mask].cpu()

                        # Get original graph ID from batch
                        graph_id = int(batch.graph_id[b].item())

                        # Save activations with original graph ID
                        torch.save(h2_graph, layer2_dir / f"graph_{graph_id}.pt")

                        saved_count += 1
                else:
                    # Single graph
                    graph_id = int(batch.graph_id.item())
                    torch.save(h2.cpu(), layer2_dir / f"graph_{graph_id}.pt")
                    saved_count += 1

        print(f"Saved layer2 activations for {saved_count} graphs to {output_dir}/activations/layer2/{split_name}/")


def collate_fn(batch: List[Data]) -> Data:
    """
    Custom collate function for batching PyG Data objects.

    Args:
        batch: List of Data objects

    Returns:
        Batched Data object
    """
    return Batch.from_data_list(batch)


def load_all_graphs(data_dir: str = "./virtual_graphs/data", single_motif_only: bool = True) -> List[Path]:
    """
    Load all graph paths from the data directory.

    Args:
        data_dir: Base directory containing graph data
        single_motif_only: If True, only load single-motif graphs (DEPRECATED - now loads from all_graphs)

    Returns:
        List of paths to graph pickle files, sorted by graph ID
    """
    data_path = Path(data_dir)

    # Load from all_graphs/raw_graphs directory
    all_graphs_dir = data_path / "all_graphs" / "raw_graphs"
    if all_graphs_dir.exists():
        # Load all .pkl files and sort by graph ID
        graph_paths = list(all_graphs_dir.glob("*.pkl"))
        # Sort by graph ID extracted from filename (graph_X.pkl)
        graph_paths.sort(key=lambda p: int(p.stem.split('_')[1]))
        return graph_paths

    # Fallback to old behavior if all_graphs doesn't exist
    print("Warning: all_graphs/raw_graphs not found, falling back to single_motif_graphs")
    graph_paths = []

    # Load single-motif graphs
    single_motif_dir = data_path / "single_motif_graphs"
    if single_motif_dir.exists():
        for motif_type_dir in single_motif_dir.iterdir():
            if motif_type_dir.is_dir():
                graph_paths.extend(sorted(motif_type_dir.glob("*.pkl")))

    # Load mixed-motif graphs only if single_motif_only is False
    if not single_motif_only:
        mixed_motif_dir = data_path / "mixed_motif_graphs"
        if mixed_motif_dir.exists():
            graph_paths.extend(sorted(mixed_motif_dir.glob("*.pkl")))

    return graph_paths


def _count_motif_distribution(paths: List[Path]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for path in paths:
        motif = _infer_motif_label(path)
        counts[motif] = counts.get(motif, 0) + 1
    return counts


def _print_split_stats(name: str, paths: List[Path]):
    counts = _count_motif_distribution(paths)
    motif_strings = [f"{motif}: {count}" for motif, count in sorted(counts.items())]
    details = ", ".join(motif_strings)
    print(f"{name} split -> total: {len(paths)} ({details})")


def _compute_motif_metrics(trainer: "GNNTrainer", data_loader: DataLoader) -> Dict[str, Dict[str, float]]:
    """Compute average masked MSE per motif for a given loader."""
    trainer.model.eval()
    motif_losses: Dict[str, List[float]] = defaultdict(list)
    motif_counts: Dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(trainer.device)
            pred = trainer.model(batch)
            loss_per_node = trainer.criterion(pred, batch.y)

            if hasattr(batch, 'batch'):
                batch_indices = batch.batch
            else:
                batch_indices = torch.zeros(batch.y.size(0), dtype=torch.long, device=batch.y.device)

            unique_graphs = torch.unique(batch_indices)

            for g in unique_graphs:
                node_mask = batch_indices == g
                masked_nodes = batch.mask[node_mask]
                node_losses = loss_per_node[node_mask]
                masked_losses = node_losses[masked_nodes]

                if masked_losses.numel() == 0:
                    continue

                graph_loss = masked_losses.mean().item()

                motif_label = "unknown"
                if hasattr(batch, 'motif_id'):
                    motif_tensor = batch.motif_id[g]
                    motif_idx = int(motif_tensor.item())
                    if 0 <= motif_idx < len(MOTIF_LABELS):
                        motif_label = MOTIF_LABELS[motif_idx]

                motif_losses[motif_label].append(graph_loss)
                motif_counts[motif_label] += 1

    metrics: Dict[str, Dict[str, float]] = {}
    for motif_label in sorted(motif_losses.keys()):
        losses = motif_losses[motif_label]
        metrics[motif_label] = {
            "num_graphs": motif_counts[motif_label],
            "mean_masked_mse": float(np.mean(losses)),
            "std_masked_mse": float(np.std(losses)) if len(losses) > 1 else 0.0
        }

    return metrics


def _save_json(data: Dict, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def split_data(graph_paths: List[Path], train_ratio: float = 0.8,
               val_ratio: float = 0.1, seed: int = 42,
               stratify_by_motif: bool = True,
               equal_counts_per_motif: bool = False) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split graph paths into train/val/test sets.

    Args:
        graph_paths: List of all graph paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed
        stratify_by_motif: Whether to split within motif groups
        equal_counts_per_motif: If True, enforce equal per-motif counts in each
            split (drops excess graphs per motif)

    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    rng = np.random.default_rng(seed)

    if not stratify_by_motif:
        n_graphs = len(graph_paths)
        indices = rng.permutation(n_graphs)

        n_train = int(n_graphs * train_ratio)
        n_val = int(n_graphs * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]

        train_paths = [graph_paths[i] for i in train_indices]
        val_paths = [graph_paths[i] for i in val_indices]
        test_paths = [graph_paths[i] for i in test_indices]

        return train_paths, val_paths, test_paths

    motif_groups: Dict[str, List[Path]] = {}
    for path in graph_paths:
        motif = _infer_motif_label(path)
        motif_groups.setdefault(motif, []).append(path)

    if equal_counts_per_motif and motif_groups:
        non_empty_counts = [len(paths) for paths in motif_groups.values() if paths]
        usable_count = min(non_empty_counts) if non_empty_counts else 0
    else:
        usable_count = None

    train_paths: List[Path] = []
    val_paths: List[Path] = []
    test_paths: List[Path] = []

    for motif_paths in motif_groups.values():
        if not motif_paths:
            continue

        shuffled = list(np.array(motif_paths)[rng.permutation(len(motif_paths))])

        if equal_counts_per_motif and usable_count is not None:
            shuffled = shuffled[:usable_count]

        n = len(shuffled)
        if n == 0:
            continue

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        train_paths.extend(shuffled[:n_train])
        val_paths.extend(shuffled[n_train:n_train + n_val])
        test_paths.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    train_paths = list(np.array(train_paths)[rng.permutation(len(train_paths))]) if train_paths else []
    val_paths = list(np.array(val_paths)[rng.permutation(len(val_paths))]) if val_paths else []
    test_paths = list(np.array(test_paths)[rng.permutation(len(test_paths))]) if test_paths else []

    return train_paths, val_paths, test_paths


def main():
    """Main training pipeline."""
    # Configuration
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32 
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    MASK_PROB = 0.3

    print(f"Using device: {DEVICE}")
    print(f"Loading graphs from ./virtual_graphs/data/")

    # Load and split data
    all_graph_paths = load_all_graphs(single_motif_only=True)
    print(f"Found {len(all_graph_paths)} graphs")

    if len(all_graph_paths) == 0:
        print("Error: No graphs found. Please run graph_motif_generator.py first.")
        return

    # Verify graph ID extraction
    if len(all_graph_paths) > 0:
        sample_ids = [int(p.stem.split('_')[1]) for p in all_graph_paths[:5]]
        print(f"Sample graph IDs (first 5): {sample_ids}")
        if len(all_graph_paths) >= 5:
            last_ids = [int(p.stem.split('_')[1]) for p in all_graph_paths[-5:]]
            print(f"Sample graph IDs (last 5): {last_ids}")

    train_paths, val_paths, test_paths = split_data(
        all_graph_paths,
        seed=SEED,
        stratify_by_motif=True,
        equal_counts_per_motif=False
    )
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    print("Motif distribution per split:")
    _print_split_stats("  Train", train_paths)
    _print_split_stats("  Val", val_paths)
    _print_split_stats("  Test", test_paths)

    # Extract and save graph IDs for each split
    train_graph_ids = [int(p.stem.split('_')[1]) for p in train_paths]
    val_graph_ids = [int(p.stem.split('_')[1]) for p in val_paths]
    test_graph_ids = [int(p.stem.split('_')[1]) for p in test_paths]

    _save_json({"graph_ids": train_graph_ids}, "outputs/train_graph_ids.json")
    _save_json({"graph_ids": val_graph_ids}, "outputs/val_graph_ids.json")
    _save_json({"graph_ids": test_graph_ids}, "outputs/test_graph_ids.json")
    print(f"Saved graph ID mappings to outputs/")

    # Create datasets
    train_dataset = GraphDataset(train_paths, mask_prob=MASK_PROB, seed=SEED)
    val_dataset = GraphDataset(val_paths, mask_prob=MASK_PROB, seed=SEED + 1)
    test_dataset = GraphDataset(test_paths, mask_prob=MASK_PROB, seed=SEED + 2)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn)

    # Initialize model and trainer
    model = GCNModel(input_dim=2, hidden_dim1=128, hidden_dim2=64, output_dim=1, dropout=0.2)
    trainer = GNNTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE, seed=SEED)

    # Training loop
    print("\nTraining GCN model...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_epoch = -1
    training_metrics = {
        "train_loss": [],
        "val_loss": []
    }

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        training_metrics["train_loss"].append(train_loss)
        training_metrics["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            trainer.save_model("checkpoints/gnn_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = trainer.validate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    training_metrics["best_val_loss"] = float(best_val_loss)
    training_metrics["best_epoch"] = best_epoch + 1 if best_epoch >= 0 else None
    training_metrics["test_loss"] = float(test_loss)
    _save_json(training_metrics, "outputs/training_metrics.json")
    print("Saved training metrics to outputs/training_metrics.json")

    # Extract and save activations
    print("\nExtracting layer2 (64-dim) activations...")
    print(f"Note: Activations will be saved using original graph IDs (0-4999)")
    trainer.extract_and_save_activations(train_loader, "outputs", "train")
    trainer.extract_and_save_activations(val_loader, "outputs", "val")
    trainer.extract_and_save_activations(test_loader, "outputs", "test")

    # Verify a few saved activations
    print("\nVerifying saved activations...")
    layer2_dir = Path("outputs/activations/layer2")
    if layer2_dir.exists():
        for split in ["train", "val", "test"]:
            split_dir = layer2_dir / split
            if split_dir.exists():
                saved_files = list(split_dir.glob("graph_*.pt"))
                if saved_files:
                    sample_file = saved_files[0]
                    sample_activation = torch.load(sample_file)
                    print(f"  {split}: {len(saved_files)} files, sample shape: {sample_activation.shape} (expected: [num_nodes, 64])")
                    # Verify the filename corresponds to a graph in the split
                    file_graph_id = int(sample_file.stem.split('_')[1])
                    if split == "train" and file_graph_id in train_graph_ids:
                        print(f"    ✓ Graph ID {file_graph_id} correctly found in train split")
                    elif split == "val" and file_graph_id in val_graph_ids:
                        print(f"    ✓ Graph ID {file_graph_id} correctly found in val split")
                    elif split == "test" and file_graph_id in test_graph_ids:
                        print(f"    ✓ Graph ID {file_graph_id} correctly found in test split")

    motif_metrics = {
        "train": _compute_motif_metrics(trainer, train_eval_loader),
        "val": _compute_motif_metrics(trainer, val_loader),
        "test": _compute_motif_metrics(trainer, test_loader)
    }
    _save_json(motif_metrics, "outputs/motif_metrics.json")
    print("Saved motif metrics to outputs/motif_metrics.json")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
