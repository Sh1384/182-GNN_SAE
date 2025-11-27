"""
Graph Motif Generator for GNN Interpretability Experiments

Generates synthetic directed weighted graphs containing various network motifs
for testing and interpreting graph neural networks.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


class GraphMotifGenerator:
    """
    Generates synthetic directed graphs with network motifs for interpretability experiments.

    Supports two categories:
    1. Single-motif graphs: Each graph contains exactly one motif type
    2. Mixed-motif graphs: Each graph contains 2-3 randomly embedded motifs

    Attributes:
        base_dir (Path): Base directory for saving generated graphs
        rng (np.random.Generator): Random number generator for reproducibility
    """

    def __init__(self, base_dir: str = "./data", seed: int = 42):
        """
        Initialize the graph generator.

        Args:
            base_dir: Base directory for saving generated graphs
            seed: Random seed for reproducibility
        """
        self.base_dir = Path(base_dir)
        self.rng = np.random.default_rng(seed)
        self.motif_types = [
            "feedforward_loop",
            "feedback_loop",
            "single_input_module",
            "cascade"
        ]

    def _sample_weight(self) -> float:
        """Sample a random edge weight from uniform distribution [-1, 1]."""
        return self.rng.uniform(-1.0, 1.0)

    def _build_feedforward_loop(self) -> nx.DiGraph:
        """
        Build a feedforward loop motif: A→B, A→C, B→C

        Returns:
            DiGraph with 3 nodes forming a feedforward loop
        """
        G = nx.DiGraph()
        nodes = self.rng.choice(range(100), size=3, replace=False)
        A, B, C = nodes

        G.add_node(A, label=f"node_{A}")
        G.add_node(B, label=f"node_{B}")
        G.add_node(C, label=f"node_{C}")

        G.add_edge(A, B, weight=self._sample_weight())
        G.add_edge(A, C, weight=self._sample_weight())
        G.add_edge(B, C, weight=self._sample_weight())

        return G

    def _build_feedback_loop(self) -> nx.DiGraph:
        """
        Build a feedback loop motif: X→Y, Y→X

        Returns:
            DiGraph with 2 nodes forming a feedback loop
        """
        G = nx.DiGraph()
        nodes = self.rng.choice(range(100), size=2, replace=False)
        X, Y = nodes

        G.add_node(X, label=f"node_{X}")
        G.add_node(Y, label=f"node_{Y}")

        G.add_edge(X, Y, weight=self._sample_weight())
        G.add_edge(Y, X, weight=self._sample_weight())

        return G

    def _build_single_input_module(self) -> nx.DiGraph:
        """
        Build a single input module: R→G1, R→G2, R→G3, possibly R→G4

        Returns:
            DiGraph with 4-5 nodes forming a single input module
        """
        G = nx.DiGraph()
        n_targets = self.rng.integers(3, 5)  # 3 or 4 target nodes
        nodes = self.rng.choice(range(100), size=n_targets + 1, replace=False)
        R = nodes[0]
        targets = nodes[1:]

        G.add_node(R, label=f"node_{R}")
        for target in targets:
            G.add_node(target, label=f"node_{target}")
            G.add_edge(R, target, weight=self._sample_weight())

        return G

    def _build_cascade(self) -> nx.DiGraph:
        """
        Build a cascade motif: A→B→C→D

        Returns:
            DiGraph with 4 nodes forming a linear cascade
        """
        G = nx.DiGraph()
        nodes = self.rng.choice(range(100), size=4, replace=False)

        for node in nodes:
            G.add_node(node, label=f"node_{node}")

        for i in range(len(nodes) - 1):
            G.add_edge(nodes[i], nodes[i+1], weight=self._sample_weight())

        return G

    def _expand_graph(self, G: nx.DiGraph, target_size: int) -> nx.DiGraph:
        """
        Expand a motif graph to target size by adding isolated nodes.

        Args:
            G: Input graph
            target_size: Desired number of nodes

        Returns:
            Expanded graph with target_size nodes
        """
        current_nodes = set(G.nodes())
        current_size = len(current_nodes)

        if current_size >= target_size:
            return G

        # Add additional isolated nodes
        available_nodes = set(range(100)) - current_nodes
        new_nodes = self.rng.choice(list(available_nodes),
                                     size=target_size - current_size,
                                     replace=False)

        for node in new_nodes:
            G.add_node(node, label=f"node_{node}")

        return G

    def _relabel_nodes_sequentially(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Relabel graph nodes to sequential integers starting from 0.

        Args:
            G: Input graph with arbitrary node labels

        Returns:
            Graph with nodes relabeled 0, 1, 2, ...
        """
        mapping = {old: new for new, old in enumerate(sorted(G.nodes()))}
        G_relabeled = nx.relabel_nodes(G, mapping, copy=True)

        # Update node labels
        for node in G_relabeled.nodes():
            G_relabeled.nodes[node]['label'] = f"node_{node}"

        return G_relabeled

    def generate_single_motif_graph(self, motif_type: str,
                                     target_size: Optional[int] = None) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Generate a 10-node graph with multiple instances of one motif type.

        Args:
            motif_type: Type of motif to generate
            target_size: Target number of nodes (always 10, parameter kept for compatibility)

        Returns:
            Tuple of (graph, metadata_df)
            - graph: DiGraph containing multiple instances of the specified motif
            - metadata_df: DataFrame with one-hot encoded motif membership per node
        """
        TARGET_SIZE = 10
        G_combined = nx.DiGraph()
        node_to_motifs = {}
        node_id = 0

        # Keep adding complete motif instances until we can't fit more
        while node_id < TARGET_SIZE:
            # Build one instance of the motif
            if motif_type == "feedforward_loop":
                motif_instance = self._build_feedforward_loop()
            elif motif_type == "feedback_loop":
                motif_instance = self._build_feedback_loop()
            elif motif_type == "single_input_module":
                motif_instance = self._build_single_input_module()
            elif motif_type == "cascade":
                motif_instance = self._build_cascade()
            else:
                raise ValueError(f"Unknown motif type: {motif_type}")

            motif_size = len(motif_instance.nodes())

            # Check if adding this instance would exceed 10 nodes
            if node_id + motif_size > TARGET_SIZE:
                break

            # Add nodes from this motif instance
            node_mapping = {}
            for old_node in sorted(motif_instance.nodes()):
                node_mapping[old_node] = node_id
                G_combined.add_node(node_id, label=f"node_{node_id}")
                node_to_motifs[node_id] = motif_type
                node_id += 1

            # Add edges from this motif instance
            for u, v in motif_instance.edges():
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                G_combined.add_edge(new_u, new_v, weight=self._sample_weight())

        # Pad with isolated nodes to reach exactly 10
        while node_id < TARGET_SIZE:
            G_combined.add_node(node_id, label=f"node_{node_id}")
            node_to_motifs[node_id] = None
            node_id += 1

        # Create metadata DataFrame
        n_nodes = TARGET_SIZE
        metadata = {mt: [0] * n_nodes for mt in self.motif_types}

        for node_id, node_motif in node_to_motifs.items():
            if node_motif is not None:
                metadata[node_motif][node_id] = 1

        metadata_df = pd.DataFrame(metadata, index=[f'node_{i}' for i in range(n_nodes)])

        # Store motif type as graph attribute
        G_combined.graph['motif_type'] = motif_type

        return G_combined, metadata_df

    def generate_single_motif_graphs(self, n_per_type: int = 1000,
                                    start_idx: int = 0) -> Tuple[List[nx.DiGraph], int]:
        """
        Generate single-motif graphs for all motif types.

        Args:
            n_per_type: Number of graphs to generate per motif type
            start_idx: Starting graph ID for sequential numbering

        Returns:
            Tuple of (list of graphs, next_graph_idx)
        """
        print("Generating single-motif graphs...")

        # Create output directories
        raw_graphs_dir = self.base_dir / "all_graphs" / "raw_graphs"
        metadata_dir = self.base_dir / "all_graphs" / "graph_motif_metadata"
        raw_graphs_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        all_graphs = []
        graph_idx = start_idx

        for motif_type in self.motif_types:
            for i in tqdm(range(n_per_type), desc=f"Generating {motif_type}"):
                G, metadata_df = self.generate_single_motif_graph(motif_type)

                # Save graph
                graph_path = raw_graphs_dir / f"graph_{graph_idx}.pkl"
                with open(graph_path, 'wb') as f:
                    pickle.dump(G, f)

                # Save metadata CSV
                metadata_path = metadata_dir / f"graph_{graph_idx}_metadata.csv"
                metadata_df.to_csv(metadata_path)

                all_graphs.append(G)
                graph_idx += 1

        print(f"Generated {n_per_type * len(self.motif_types)} single-motif graphs (IDs {start_idx}-{graph_idx-1})")
        return all_graphs, graph_idx

    def _merge_motifs(self, motifs: List[nx.DiGraph],
                      motif_types: List[str],
                      target_size: int,
                      extra_edge_prob: float = 0.25) -> Tuple[nx.DiGraph, Dict[int, List[str]]]:
        """
        DEPRECATED: This method is no longer used as graph generation now creates
        10-node graphs directly in generate_single_motif_graph() and
        generate_mixed_motif_graph(). Kept for backward compatibility.

        Merge multiple motifs into a single graph with random interconnections.

        Args:
            motifs: List of motif graphs to merge
            motif_types: List of motif type names corresponding to each motif
            target_size: Target number of nodes for final graph
            extra_edge_prob: Probability of adding extra random edges

        Returns:
            Tuple of (merged_graph, node_motif_mapping)
            - merged_graph: Combined graph
            - node_motif_mapping: Dict mapping old node IDs to list of motif types
        """
        # Create combined graph
        G_combined = nx.DiGraph()
        node_offset = 0
        node_to_motifs = {}

        # Add all motifs with offset node labels
        for motif, motif_type in zip(motifs, motif_types):
            mapping = {old: old + node_offset for old in motif.nodes()}
            motif_relabeled = nx.relabel_nodes(motif, mapping)

            # Track which nodes belong to which motifs (before relabeling)
            for old_node in motif.nodes():
                new_node = mapping[old_node]
                if new_node not in node_to_motifs:
                    node_to_motifs[new_node] = []
                node_to_motifs[new_node].append(motif_type)

            G_combined = nx.compose(G_combined, motif_relabeled)
            node_offset += len(motif.nodes())

        # Expand to target size if needed
        current_size = len(G_combined.nodes())
        if current_size < target_size:
            for i in range(current_size, target_size):
                G_combined.add_node(i, label=f"node_{i}")
                node_to_motifs[i] = []  # Isolated nodes don't belong to any motif

        # Add random interconnections between motifs
        nodes = list(G_combined.nodes())
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i != j and not G_combined.has_edge(nodes[i], nodes[j]):
                    if self.rng.random() < extra_edge_prob:
                        G_combined.add_edge(nodes[i], nodes[j],
                                           weight=self._sample_weight())

        # Create mapping for relabeled nodes
        old_nodes = sorted(G_combined.nodes())
        node_mapping = {old: new for new, old in enumerate(old_nodes)}

        # Update node_to_motifs with new labels
        node_to_motifs_relabeled = {node_mapping[old]: motifs
                                     for old, motifs in node_to_motifs.items()}

        # Relabel sequentially
        G_combined = self._relabel_nodes_sequentially(G_combined)

        return G_combined, node_to_motifs_relabeled

    def generate_mixed_motif_graph(self) -> Tuple[nx.DiGraph, pd.DataFrame]:
        """
        Generate a 10-node graph with instances of different motif types.

        Returns:
            Tuple of (graph, metadata_df)
            - graph: DiGraph containing multiple motif instances with random interconnections
            - metadata_df: DataFrame with one-hot encoded motif membership per node
        """
        TARGET_SIZE = 10

        # Choose 2-3 different motif types
        n_motif_types = self.rng.integers(2, 4)
        chosen_motifs = self.rng.choice(self.motif_types, size=n_motif_types, replace=True)

        G_combined = nx.DiGraph()
        node_to_motifs = {}
        node_id = 0

        # Add instances of each chosen motif type
        for motif_type in chosen_motifs:
            if node_id >= TARGET_SIZE:
                break

            # Build one instance
            if motif_type == "feedforward_loop":
                motif_instance = self._build_feedforward_loop()
            elif motif_type == "feedback_loop":
                motif_instance = self._build_feedback_loop()
            elif motif_type == "single_input_module":
                motif_instance = self._build_single_input_module()
            elif motif_type == "cascade":
                motif_instance = self._build_cascade()
            else:
                raise ValueError(f"Unknown motif type: {motif_type}")

            motif_size = len(motif_instance.nodes())

            # Check if it fits
            if node_id + motif_size > TARGET_SIZE:
                break

            # Add nodes and edges
            node_mapping = {}
            for old_node in sorted(motif_instance.nodes()):
                node_mapping[old_node] = node_id
                G_combined.add_node(node_id, label=f"node_{node_id}")
                node_to_motifs[node_id] = motif_type
                node_id += 1

            for u, v in motif_instance.edges():
                new_u = node_mapping[u]
                new_v = node_mapping[v]
                G_combined.add_edge(new_u, new_v, weight=self._sample_weight())

        # Pad with isolated nodes
        while node_id < TARGET_SIZE:
            G_combined.add_node(node_id, label=f"node_{node_id}")
            node_to_motifs[node_id] = None
            node_id += 1

        # Add random interconnections (20-30% probability)
        extra_edge_prob = self.rng.uniform(0.2, 0.3)
        nodes = list(G_combined.nodes())
        for i in nodes:
            for j in nodes:
                if i != j and not G_combined.has_edge(i, j):
                    if self.rng.random() < extra_edge_prob:
                        G_combined.add_edge(i, j, weight=self._sample_weight())

        # Create metadata
        metadata = {mt: [0] * TARGET_SIZE for mt in self.motif_types}
        for node_id, node_motif in node_to_motifs.items():
            if node_motif is not None:
                metadata[node_motif][node_id] = 1

        metadata_df = pd.DataFrame(metadata, index=[f'node_{i}' for i in range(TARGET_SIZE)])

        # Store motif composition
        G_combined.graph['motif_composition'] = list(chosen_motifs)

        return G_combined, metadata_df

    def generate_mixed_motif_graphs(self, n_graphs: int = 1000,
                                   start_idx: int = 0) -> Tuple[List[nx.DiGraph], int]:
        """
        Generate mixed-motif graphs with metadata.

        Args:
            n_graphs: Number of mixed-motif graphs to generate
            start_idx: Starting graph ID for sequential numbering

        Returns:
            Tuple of (list of graphs, next_graph_idx)
        """
        print("Generating mixed-motif graphs...")

        # Create output directories
        raw_graphs_dir = self.base_dir / "all_graphs" / "raw_graphs"
        metadata_dir = self.base_dir / "all_graphs" / "graph_motif_metadata"
        raw_graphs_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        graphs = []
        graph_idx = start_idx

        for i in tqdm(range(n_graphs), desc="Generating mixed motifs"):
            G, metadata_df = self.generate_mixed_motif_graph()

            # Save graph
            graph_path = raw_graphs_dir / f"graph_{graph_idx}.pkl"
            with open(graph_path, 'wb') as f:
                pickle.dump(G, f)

            # Save metadata CSV
            metadata_path = metadata_dir / f"graph_{graph_idx}_metadata.csv"
            metadata_df.to_csv(metadata_path)

            graphs.append(G)
            graph_idx += 1

        print(f"Generated {n_graphs} mixed-motif graphs (IDs {start_idx}-{graph_idx-1})")
        return graphs, graph_idx

    def simulate_expression(self, W: np.ndarray,
                           steps: int = 50,
                           gamma: float = 0.3,
                           noise_std: float = 0.01,
                           return_trajectory: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simulate gene expression dynamics using iterative update rule.

        The update rule is:
        x_{t+1} = (1-γ)x_t + γ·sigmoid(Wx_t) + ε

        Args:
            W: Weighted adjacency matrix (n x n numpy array)
            steps: Number of simulation steps
            gamma: Update rate parameter
            noise_std: Standard deviation of Gaussian noise
            return_trajectory: If True, return full time series

        Returns:
            Tuple of (final_expression, trajectory)
            - final_expression: Expression values at final timestep (n,)
            - trajectory: Full time series if return_trajectory=True, else None (steps, n)
        """
        n_nodes = W.shape[0]

        # Initialize with random expression values
        x = self.rng.uniform(0, 1, size=n_nodes)

        if return_trajectory:
            trajectory = np.zeros((steps, n_nodes))
            trajectory[0] = x

        # Simulate dynamics
        for t in range(1, steps):
            # Compute sigmoid of weighted inputs
            weighted_input = W @ x
            sigmoid_input = 1.0 / (1.0 + np.exp(-weighted_input))

            # Add noise
            noise = self.rng.normal(0, noise_std, size=n_nodes)

            # Update expression
            x = (1 - gamma) * x + gamma * sigmoid_input + noise

            # Clip to valid range
            x = np.clip(x, 0, 1)

            if return_trajectory:
                trajectory[t] = x

        if return_trajectory:
            return x, trajectory
        else:
            return x, None

    def _load_graphs_from_dir(self, directory: Path) -> List[Tuple[Path, nx.DiGraph]]:
        """
        Load all graphs from a directory.

        Args:
            directory: Path to directory containing pickled graphs

        Returns:
            List of (filepath, graph) tuples
        """
        graphs = []
        for pkl_file in sorted(directory.glob("**/*.pkl")):
            with open(pkl_file, 'rb') as f:
                G = pickle.load(f)
                graphs.append((pkl_file, G))
        return graphs

    def simulate_all(self, output_dir: str = "./data/simulations",
                     steps: int = 50,
                     gamma: float = 0.3,
                     noise_std: float = 0.01) -> None:
        """
        Run expression simulation for all saved graphs and store results.

        Args:
            output_dir: Directory to save simulation results
            steps: Number of simulation steps
            gamma: Update rate parameter
            noise_std: Standard deviation of Gaussian noise
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("Simulating expression dynamics for all graphs...")

        # Process all graphs from the unified directory
        raw_graphs_dir = self.base_dir / "all_graphs" / "raw_graphs"
        if not raw_graphs_dir.exists():
            print(f"Error: No graphs found in {raw_graphs_dir}")
            return

        graphs = self._load_graphs_from_dir(raw_graphs_dir)

        for pkl_path, G in tqdm(graphs, desc="Simulating all graphs"):
            # Extract adjacency matrix
            n_nodes = len(G.nodes())
            W = nx.to_numpy_array(G, weight='weight')

            # Extract node labels
            node_labels = [G.nodes[i].get('label', f'node_{i}')
                          for i in range(n_nodes)]

            # Simulate expression
            final_expression, trajectory = self.simulate_expression(
                W, steps=steps, gamma=gamma, noise_std=noise_std,
                return_trajectory=True
            )

            # Save results
            output_file = output_path / f"{pkl_path.stem}_sim.npz"
            np.savez(output_file,
                    adjacency=W,
                    node_labels=node_labels,
                    expression=final_expression,
                    trajectory=trajectory)

        print(f"Simulation results saved to {output_dir}")


def main():
    """Example usage of GraphMotifGenerator."""
    # Initialize generator
    generator = GraphMotifGenerator(base_dir="./data", seed=42)

    # Generate single-motif graphs (starting from index 0)
    single_graphs, next_idx = generator.generate_single_motif_graphs(n_per_type=1000, start_idx=0)

    # Generate mixed-motif graphs (continuing from where single-motif ended)
    mixed_graphs, final_idx = generator.generate_mixed_motif_graphs(n_graphs=1000, start_idx=next_idx)

    print(f"\nTotal graphs generated: {final_idx}")
    print(f"  - Single-motif: 0 to {next_idx-1}")
    print(f"  - Mixed-motif: {next_idx} to {final_idx-1}")
    print(f"\nAll graphs saved to: ./data/all_graphs/raw_graphs/")
    print(f"All metadata saved to: ./data/all_graphs/graph_motif_metadata/")

    # Run simulations on all graphs
    generator.simulate_all(output_dir="./data/simulations", steps=50)

    print("\nAll graphs generated and simulated successfully!")


if __name__ == "__main__":
    main()
