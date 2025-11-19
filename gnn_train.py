"""
GNN Training Module for Node Value Prediction

Trains a two-layer Graph Convolutional Network (GCN) to predict missing node values
from partially observed synthetic graphs. Saves trained models and layer activations
for downstream interpretability analysis.
"""

import os
import pickle
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


class GraphDataset(Dataset):
    """
    Dataset for loading synthetic graphs with expression data.

    Attributes:
        graphs: List of tuples (graph_path, expression_data)
        mask_prob: Probability of masking each node during training
        rng: Random number generator
    """

    def __init__(self, graph_paths: List[Path], mask_prob: float = 0.3, seed: int = 42):
        """
        Initialize the graph dataset.

        Args:
            graph_paths: List of paths to pickled graph files
            mask_prob: Probability of masking nodes for prediction
            seed: Random seed for reproducibility
        """
        self.graph_paths = graph_paths
        self.mask_prob = mask_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.graph_paths)

    def __getitem__(self, idx: int) -> Data:
        """
        Load a graph and simulate expression values.

        Args:
            idx: Index of graph to load

        Returns:
            PyG Data object with node features, edge indices, and labels
        """
        # Load graph
        with open(self.graph_paths[idx], 'rb') as f:
            G = pickle.load(f)

        # Get adjacency matrix and simulate expression
        import networkx as nx
        n_nodes = len(G.nodes())
        W = nx.to_numpy_array(G, weight='weight')

        # Simulate expression dynamics
        expression = self._simulate_expression(W)

        # Create mask for training (which nodes to predict)
        mask = self.rng.random(n_nodes) < self.mask_prob
        mask = torch.tensor(mask, dtype=torch.bool)

        # Create node features: [normalized_expression, mask_flag]
        # For masked nodes, set expression to 0
        expression_tensor = torch.tensor(expression, dtype=torch.float32)
        masked_expression = expression_tensor.clone()
        masked_expression[mask] = 0.0

        # Normalize to [0, 1] range
        if masked_expression.max() > 0:
            masked_expression = masked_expression / (masked_expression.max() + 1e-8)

        # Stack features
        mask_flag = (~mask).float().unsqueeze(1)  # 1 if observed, 0 if masked
        x = torch.cat([masked_expression.unsqueeze(1), mask_flag], dim=1)

        # Create edge index from adjacency matrix
        edge_index = torch.tensor(np.array(np.nonzero(W)), dtype=torch.long)
        edge_weight = torch.tensor(W[W != 0], dtype=torch.float32)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=expression_tensor,
            mask=mask,
            num_nodes=n_nodes
        )

        return data

    def _simulate_expression(self, W: np.ndarray, steps: int = 50,
                            gamma: float = 0.3, noise_std: float = 0.01) -> np.ndarray:
        """
        Simulate gene expression dynamics.

        Args:
            W: Weighted adjacency matrix
            steps: Number of simulation steps
            gamma: Update rate parameter
            noise_std: Standard deviation of Gaussian noise

        Returns:
            Final expression values
        """
        n_nodes = W.shape[0]
        x = self.rng.uniform(0, 1, size=n_nodes)

        for _ in range(steps):
            weighted_input = W @ x
            sigmoid_input = 1.0 / (1.0 + np.exp(-np.clip(weighted_input, -10, 10)))
            noise = self.rng.normal(0, noise_std, size=n_nodes)
            x = (1 - gamma) * x + gamma * sigmoid_input + noise
            x = np.clip(x, 0, 1)

        return x


class GCNModel(nn.Module):
    """
    Two-layer Graph Convolutional Network for node value prediction.

    Architecture:
        - Layer 1: GCNConv(2 -> 64) + ReLU + Dropout
        - Layer 2: GCNConv(64 -> 1)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64,
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the GCN model.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(GCNModel, self).__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

        # Storage for activations
        self.layer1_activations = None
        self.layer2_activations = None

    def forward(self, data: Data, store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass through the GCN.

        Args:
            data: PyG Data object
            store_activations: Whether to store layer activations

        Returns:
            Predicted node values
        """
        x, edge_index = data.x, data.edge_index
        edge_weight = getattr(data, 'edge_attr', None)

        # Layer 1: GCNConv + ReLU + Dropout
        h1 = self.conv1(x, edge_index, edge_weight=edge_weight)
        h1 = F.relu(h1)

        if store_activations:
            self.layer1_activations = h1.detach()

        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2: GCNConv
        h2 = self.conv2(h1, edge_index, edge_weight=edge_weight)

        if store_activations:
            self.layer2_activations = h2.detach()

        return h2.squeeze(-1)

    def get_activations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get stored layer activations.

        Returns:
            Tuple of (layer1_activations, layer2_activations)
        """
        return self.layer1_activations, self.layer2_activations


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
        Extract and save layer activations for all graphs.

        Args:
            data_loader: DataLoader for graphs
            output_dir: Base output directory
            split_name: Name of data split (train/val/test)
        """
        self.model.eval()

        layer1_dir = Path(output_dir) / "activations" / "layer1" / split_name
        layer2_dir = Path(output_dir) / "activations" / "layer2" / split_name
        layer1_dir.mkdir(parents=True, exist_ok=True)
        layer2_dir.mkdir(parents=True, exist_ok=True)

        graph_idx = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {split_name} activations"):
                batch = batch.to(self.device)

                # Forward pass with activation storage
                _ = self.model(batch, store_activations=True)
                h1, h2 = self.model.get_activations()

                # Split batch back into individual graphs
                if hasattr(batch, 'batch'):
                    # Handle batched data
                    batch_indices = batch.batch
                    unique_batches = torch.unique(batch_indices)

                    for b in unique_batches:
                        mask = batch_indices == b

                        h1_graph = h1[mask].cpu()
                        h2_graph = h2[mask].cpu()

                        # Save activations
                        torch.save(h1_graph, layer1_dir / f"graph_{graph_idx}.pt")
                        torch.save(h2_graph, layer2_dir / f"graph_{graph_idx}.pt")

                        graph_idx += 1
                else:
                    # Single graph
                    torch.save(h1.cpu(), layer1_dir / f"graph_{graph_idx}.pt")
                    torch.save(h2.cpu(), layer2_dir / f"graph_{graph_idx}.pt")
                    graph_idx += 1

        print(f"Saved activations for {graph_idx} graphs to {output_dir}/activations/")


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
        single_motif_only: If True, only load single-motif graphs

    Returns:
        List of paths to graph pickle files
    """
    data_path = Path(data_dir)
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


def split_data(graph_paths: List[Path], train_ratio: float = 0.8,
               val_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split graph paths into train/val/test sets.

    Args:
        graph_paths: List of all graph paths
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    rng = np.random.default_rng(seed)
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

    train_paths, val_paths, test_paths = split_data(all_graph_paths, seed=SEED)
    print(f"Split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")

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

    # Initialize model and trainer
    model = GCNModel(input_dim=2, hidden_dim=64, output_dim=1, dropout=0.2)
    trainer = GNNTrainer(model, device=DEVICE, learning_rate=LEARNING_RATE, seed=SEED)

    # Training loop
    print("\nTraining GCN model...")
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in tqdm(range(NUM_EPOCHS), desc="Training"):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.validate(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
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

    # Extract and save activations
    print("\nExtracting activations...")
    trainer.extract_and_save_activations(train_loader, "outputs", "train")
    trainer.extract_and_save_activations(val_loader, "outputs", "val")
    trainer.extract_and_save_activations(test_loader, "outputs", "test")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
