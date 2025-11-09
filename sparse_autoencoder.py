"""
Sparse Autoencoder Module for GNN Activation Analysis

Trains sparse autoencoders on frozen GNN layer activations to discover
interpretable features that correlate with network motifs.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for discovering interpretable features in GNN activations.

    Architecture:
        - Encoder: Linear(input_dim -> latent_dim) + ReLU
        - Decoder: Linear(latent_dim -> input_dim)

    Loss:
        L = ||x - x_hat||_2^2 + lambda * ||z||_1
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 512,
                 sparsity_lambda: float = 1e-3):
        """
        Initialize the sparse autoencoder.

        Args:
            input_dim: Dimension of input activations
            latent_dim: Dimension of latent representation
            sparsity_lambda: Sparsity penalty weight
        """
        super(SparseAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparsity_lambda = sparsity_lambda

        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim)

        # Decoder
        self.decoder = nn.Linear(latent_dim, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Latent representation (batch_size, latent_dim)
        """
        z = self.encoder(x)
        z = F.relu(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            z: Latent representation (batch_size, latent_dim)

        Returns:
            Reconstructed activations (batch_size, input_dim)
        """
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input activations (batch_size, input_dim)

        Returns:
            Tuple of (reconstructed_x, latent_z)
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, x: torch.Tensor, x_hat: torch.Tensor,
             z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss with reconstruction and sparsity terms.

        Args:
            x: Original input
            x_hat: Reconstructed input
            z: Latent representation

        Returns:
            Tuple of (total_loss, reconstruction_loss, sparsity_loss)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_hat, x)

        # Sparsity loss (L1 penalty on latent)
        sparsity_loss = torch.mean(torch.abs(z))

        # Total loss
        total_loss = recon_loss + self.sparsity_lambda * sparsity_loss

        return total_loss, recon_loss, sparsity_loss


class SAETrainer:
    """
    Trainer for Sparse Autoencoder on GNN activations.

    Handles loading activations, training, and saving latent features.
    """

    def __init__(self, model: SparseAutoencoder, device: str = 'cuda',
                 learning_rate: float = 1e-3, seed: int = 42):
        """
        Initialize the SAE trainer.

        Args:
            model: Sparse autoencoder model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            seed: Random seed
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: DataLoader for training data

        Returns:
            Tuple of (total_loss, recon_loss, sparsity_loss)
        """
        self.model.train()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        sparsity_loss_sum = 0.0
        num_batches = 0

        for batch in train_loader:
            x = batch[0].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            x_hat, z = self.model(x)

            # Compute loss
            total_loss, recon_loss, sparsity_loss = self.model.loss(x, x_hat, z)

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            sparsity_loss_sum += sparsity_loss.item()
            num_batches += 1

        return (total_loss_sum / num_batches,
                recon_loss_sum / num_batches,
                sparsity_loss_sum / num_batches)

    def validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate the model.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Tuple of (total_loss, recon_loss, sparsity_loss)
        """
        self.model.eval()
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        sparsity_loss_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(self.device)

                # Forward pass
                x_hat, z = self.model(x)

                # Compute loss
                total_loss, recon_loss, sparsity_loss = self.model.loss(x, x_hat, z)

                total_loss_sum += total_loss.item()
                recon_loss_sum += recon_loss.item()
                sparsity_loss_sum += sparsity_loss.item()
                num_batches += 1

        return (total_loss_sum / num_batches,
                recon_loss_sum / num_batches,
                sparsity_loss_sum / num_batches)

    def save_model(self, path: str):
        """
        Save model weights.

        Args:
            path: Path to save model
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.input_dim,
            'latent_dim': self.model.latent_dim,
            'sparsity_lambda': self.model.sparsity_lambda
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """
        Load model weights.

        Args:
            path: Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")

    def extract_and_save_latents(self, data_loader: DataLoader,
                                 output_dir: str, split_name: str):
        """
        Extract and save latent representations.

        Args:
            data_loader: DataLoader for activations
            output_dir: Output directory
            split_name: Name of data split
        """
        self.model.eval()

        output_path = Path(output_dir) / split_name
        output_path.mkdir(parents=True, exist_ok=True)

        all_latents = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Extracting {split_name} latents"):
                x = batch[0].to(self.device)
                z = self.model.encode(x)
                all_latents.append(z.cpu())

        # Concatenate and save
        all_latents = torch.cat(all_latents, dim=0)

        # Save as individual node latents
        for i in range(len(all_latents)):
            save_path = output_path / f"node_{i}.pt"
            torch.save(all_latents[i], save_path)

        # Also save concatenated version
        torch.save(all_latents, output_path / "all_latents.pt")

        print(f"Saved {len(all_latents)} latent representations to {output_dir}/{split_name}/")


def load_activations(activation_dir: Path, split_name: str) -> torch.Tensor:
    """
    Load all activations from a directory and concatenate.

    Args:
        activation_dir: Directory containing activation files
        split_name: Name of data split (train/val/test)

    Returns:
        Tensor of all activations (num_nodes, activation_dim)
    """
    activation_path = activation_dir / split_name
    if not activation_path.exists():
        raise ValueError(f"Activation directory not found: {activation_path}")

    activation_files = sorted(activation_path.glob("graph_*.pt"))

    if len(activation_files) == 0:
        raise ValueError(f"No activation files found in {activation_path}")

    all_activations = []

    for act_file in activation_files:
        act = torch.load(act_file, map_location='cpu')
        # Handle both single graph and batched cases
        if act.dim() == 1:
            act = act.unsqueeze(0)
        all_activations.append(act)

    # Concatenate all activations
    activations = torch.cat(all_activations, dim=0)

    return activations


def train_sae_for_layer(layer_name: str, layer_idx: int,
                        input_dim: int = 64,
                        latent_dim: int = 512,
                        sparsity_lambda: float = 1e-3,
                        num_epochs: int = 100,
                        batch_size: int = 256,
                        learning_rate: float = 1e-3,
                        device: str = 'cuda',
                        seed: int = 42):
    """
    Train a sparse autoencoder for a specific GNN layer.

    Args:
        layer_name: Name of layer (e.g., 'layer1', 'layer2')
        layer_idx: Index of layer (1 or 2)
        input_dim: Dimension of layer activations
        latent_dim: Dimension of latent space
        sparsity_lambda: Sparsity penalty weight
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to train on
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Training SAE for {layer_name} (dim={input_dim} -> {latent_dim})")
    print(f"{'='*60}")

    # Load activations
    activation_dir = Path("outputs/activations") / layer_name

    print(f"Loading activations from {activation_dir}...")
    train_activations = load_activations(activation_dir, "train")
    val_activations = load_activations(activation_dir, "val")
    test_activations = load_activations(activation_dir, "test")

    print(f"Train activations: {train_activations.shape}")
    print(f"Val activations: {val_activations.shape}")
    print(f"Test activations: {test_activations.shape}")

    # Create datasets and loaders
    train_dataset = TensorDataset(train_activations)
    val_dataset = TensorDataset(val_activations)
    test_dataset = TensorDataset(test_activations)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False)

    # Initialize model and trainer
    model = SparseAutoencoder(input_dim=input_dim,
                             latent_dim=latent_dim,
                             sparsity_lambda=sparsity_lambda)
    trainer = SAETrainer(model, device=device,
                        learning_rate=learning_rate, seed=seed)

    # Training loop
    print(f"\nTraining SAE for {layer_name}...")
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in tqdm(range(num_epochs), desc=f"Training {layer_name}"):
        train_loss, train_recon, train_sparsity = trainer.train_epoch(train_loader)
        val_loss, val_recon, val_sparsity = trainer.validate(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, Sparsity: {train_sparsity:.4f}")
            print(f"  Val   - Total: {val_loss:.4f}, Recon: {val_recon:.4f}, Sparsity: {val_sparsity:.4f}")

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            trainer.save_model(f"checkpoints/sae_{layer_name}.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluate on test set
    print(f"\nEvaluating {layer_name} on test set...")
    test_loss, test_recon, test_sparsity = trainer.validate(test_loader)
    print(f"Test - Total: {test_loss:.4f}, Recon: {test_recon:.4f}, Sparsity: {test_sparsity:.4f}")

    # Extract and save latent features
    print(f"\nExtracting latent features for {layer_name}...")
    output_dir = f"outputs/sae_latents/{layer_name}"

    trainer.extract_and_save_latents(train_loader, output_dir, "train")
    trainer.extract_and_save_latents(val_loader, output_dir, "val")
    trainer.extract_and_save_latents(test_loader, output_dir, "test")

    print(f"\n{layer_name} training complete!")


def main():
    """Main training pipeline for SAEs."""
    # Configuration
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 100
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    SPARSITY_LAMBDA = 1e-3

    print(f"Using device: {DEVICE}")

    # Check if activations exist
    if not Path("outputs/activations").exists():
        print("Error: No activations found. Please run gnn_train.py first.")
        return

    # Train SAE for Layer 1 (64-dim activations)
    train_sae_for_layer(
        layer_name="layer1",
        layer_idx=1,
        input_dim=64,
        latent_dim=512,
        sparsity_lambda=SPARSITY_LAMBDA,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        seed=SEED
    )

    # Train SAE for Layer 2 (1-dim activations -> use smaller latent dim)
    train_sae_for_layer(
        layer_name="layer2",
        layer_idx=2,
        input_dim=1,
        latent_dim=32,  # Smaller latent dim for 1-d input
        sparsity_lambda=SPARSITY_LAMBDA,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        seed=SEED + 1
    )

    print("\n" + "="*60)
    print("All SAE training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
