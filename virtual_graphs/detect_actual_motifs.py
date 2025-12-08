"""
Motif Detection Module

Detects actual network motifs present in graph structure through
pattern matching. Returns multi-label node assignments.

Motif Types:
- Feedback Loop: X↔Y (bidirectional)
- Feedforward Loop: A→B, A→C, B→C
- Single Input Module: R→G1, R→G2, R→G3, ... (hub)
- Cascade: A→B→C→D (linear chain)
"""

import pickle
from pathlib import Path
from typing import Set
import networkx as nx
import pandas as pd


def detect_feedback_loops(G: nx.DiGraph) -> Set[int]:
    """
    Detect feedback loop motifs (bidirectional edges).

    Pattern: X→Y and Y→X

    Args:
        G: Directed graph

    Returns:
        Set of node IDs participating in feedback loops
    """
    nodes_in_motif = set()
    for i in G.nodes():
        for j in G.nodes():
            if i < j and G.has_edge(i, j) and G.has_edge(j, i):
                nodes_in_motif.add(i)
                nodes_in_motif.add(j)
    return nodes_in_motif


def detect_feedforward_loops(G: nx.DiGraph) -> Set[int]:
    """
    Detect feedforward loop motifs.

    Pattern: A→B, A→C, B→C

    Args:
        G: Directed graph

    Returns:
        Set of node IDs participating in feedforward loops
    """
    nodes_in_motif = set()
    for a in G.nodes():
        for b in G.nodes():
            if a == b or not G.has_edge(a, b):
                continue
            for c in G.nodes():
                if c in (a, b):
                    continue
                if G.has_edge(a, c) and G.has_edge(b, c):
                    nodes_in_motif.update([a, b, c])
    return nodes_in_motif


def detect_single_input_modules(G: nx.DiGraph) -> Set[int]:
    """
    Detect single input module motifs (hub-and-spoke).

    Pattern: R→G1, R→G2, R→G3, ... (R has out-degree ≥ 3)

    Args:
        G: Directed graph

    Returns:
        Set of node IDs participating in single input modules
    """
    nodes_in_motif = set()
    for r in G.nodes():
        targets = list(G.successors(r))
        if len(targets) >= 3:
            # Check targets don't feed back to R (true fan-out)
            is_pure_fanout = all(not G.has_edge(t, r) for t in targets)
            if is_pure_fanout:
                nodes_in_motif.add(r)
                nodes_in_motif.update(targets)
    return nodes_in_motif


def detect_cascades(G: nx.DiGraph) -> Set[int]:
    """
    Detect cascade motifs (linear chains).

    Pattern: A→B→C→D (path length ≥ 4 nodes)
    Internal nodes must have in-degree=1 and out-degree=1 within path.

    Args:
        G: Directed graph

    Returns:
        Set of node IDs participating in cascades
    """
    nodes_in_motif = set()

    # Find all simple paths of length >= 4 nodes
    for source in G.nodes():
        for target in G.nodes():
            if source == target:
                continue

            # Get all paths from source to target
            try:
                paths = list(nx.all_simple_paths(G, source, target, cutoff=10))
            except nx.NetworkXNoPath:
                continue

            for path in paths:
                if len(path) >= 4:
                    # Check if it's a linear cascade
                    is_linear = True
                    for i, node in enumerate(path):
                        if i == 0 or i == len(path) - 1:
                            continue  # Skip source and target

                        # Internal nodes should have in_degree=1, out_degree=1 within path
                        in_path_predecessors = [p for p in G.predecessors(node) if p in path]
                        in_path_successors = [s for s in G.successors(node) if s in path]

                        if len(in_path_predecessors) != 1 or len(in_path_successors) != 1:
                            is_linear = False
                            break

                    if is_linear:
                        nodes_in_motif.update(path)

    return nodes_in_motif


def detect_all_motifs(graph_path: Path) -> pd.DataFrame:
    """
    Detect all motifs in a graph and return multi-label metadata.

    Args:
        graph_path: Path to pickled NetworkX graph

    Returns:
        DataFrame with binary columns for each motif type (multi-label)
    """
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)

    # Detect each motif type
    ffl_nodes = detect_feedforward_loops(G)
    fbl_nodes = detect_feedback_loops(G)
    sim_nodes = detect_single_input_modules(G)
    cas_nodes = detect_cascades(G)

    # Create multi-label metadata
    metadata = {
        'feedforward_loop': [1 if i in ffl_nodes else 0 for i in range(10)],
        'feedback_loop': [1 if i in fbl_nodes else 0 for i in range(10)],
        'single_input_module': [1 if i in sim_nodes else 0 for i in range(10)],
        'cascade': [1 if i in cas_nodes else 0 for i in range(10)]
    }

    df = pd.DataFrame(metadata, index=[f'node_{i}' for i in range(10)])
    return df
