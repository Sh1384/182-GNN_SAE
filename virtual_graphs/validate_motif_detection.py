#!/usr/bin/env python3
"""
Validate Motif Detection

Tests motif detection on sample graphs to verify correctness.
"""

import pickle
from pathlib import Path
import pandas as pd
import networkx as nx
from detect_actual_motifs import detect_all_motifs


def load_graph(graph_id: int) -> nx.DiGraph:
    """Load a graph by ID."""
    path = Path(f"data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    with open(path, 'rb') as f:
        return pickle.load(f)


def print_graph_structure(graph_id: int):
    """Print detailed structure of a graph."""
    G = load_graph(graph_id)

    print(f"\n{'='*70}")
    print(f"GRAPH {graph_id} STRUCTURE")
    print(f"{'='*70}")
    print(f"Nodes: {len(G.nodes())}")
    print(f"Edges: {len(G.edges())}")

    if 'motif_type' in G.graph:
        print(f"Motif type: {G.graph['motif_type']}")
    elif 'motif_composition' in G.graph:
        print(f"Motif composition: {G.graph['motif_composition']}")

    print("\nEdge list:")
    for u, v in sorted(G.edges()):
        print(f"  {u} → {v}")

    print("\nManual motif detection:")

    # Check for feedback loops
    print("\n  Feedback Loops (X↔Y):")
    found_fbl = False
    for i in G.nodes():
        for j in G.nodes():
            if i < j and G.has_edge(i, j) and G.has_edge(j, i):
                print(f"    {i} ↔ {j}")
                found_fbl = True
    if not found_fbl:
        print("    None")

    # Check for feedforward loops
    print("\n  Feedforward Loops (A→B, A→C, B→C):")
    found_ffl = False
    for a in G.nodes():
        for b in G.nodes():
            if a == b or not G.has_edge(a, b):
                continue
            for c in G.nodes():
                if c in (a, b):
                    continue
                if G.has_edge(a, c) and G.has_edge(b, c):
                    print(f"    {a}→{b}, {a}→{c}, {b}→{c}")
                    found_ffl = True
    if not found_ffl:
        print("    None")

    # Check for single input modules
    print("\n  Single Input Modules (R→G1,G2,G3,...):")
    found_sim = False
    for r in G.nodes():
        targets = list(G.successors(r))
        if len(targets) >= 3:
            is_pure = all(not G.has_edge(t, r) for t in targets)
            if is_pure:
                print(f"    {r} → {targets}")
                found_sim = True
    if not found_sim:
        print("    None")


def compare_old_vs_new(graph_id: int):
    """Compare old and new metadata for a graph."""
    # Load old metadata
    old_path = Path(f"data/all_graphs/graph_motif_metadata/graph_{graph_id}_metadata.csv")
    old_metadata = pd.read_csv(old_path, index_col=0)

    # Generate new metadata
    graph_path = Path(f"data/all_graphs/raw_graphs/graph_{graph_id}.pkl")
    new_metadata = detect_all_motifs(graph_path)

    print(f"\n{'='*70}")
    print(f"METADATA COMPARISON - GRAPH {graph_id}")
    print(f"{'='*70}")

    print("\nOLD METADATA (before correction):")
    print(old_metadata)
    print(f"Column sums: {old_metadata.sum().to_dict()}")

    print("\nNEW METADATA (after correction):")
    print(new_metadata)
    print(f"Column sums: {new_metadata.sum().to_dict()}")

    print("\nCHANGES:")
    changed = False
    for col in old_metadata.columns:
        if not old_metadata[col].equals(new_metadata[col]):
            changed = True
            old_nodes = set(old_metadata[old_metadata[col] == 1].index)
            new_nodes = set(new_metadata[new_metadata[col] == 1].index)
            added = new_nodes - old_nodes
            removed = old_nodes - new_nodes
            if added:
                print(f"  {col}: Added {added}")
            if removed:
                print(f"  {col}: Removed {removed}")

    if not changed:
        print("  No changes")


def main():
    print("="*70)
    print("MOTIF DETECTION VALIDATION")
    print("="*70)

    # Test graphs from each category
    test_graphs = [
        (0, "Feedforward Loop (single-motif)"),
        (1000, "Feedback Loop (single-motif)"),
        (2000, "Single Input Module (single-motif)"),
        (3000, "Cascade (single-motif)"),
        (4000, "Mixed motifs")
    ]

    for graph_id, description in test_graphs:
        print(f"\n\n{'#'*70}")
        print(f"# TEST: {description}")
        print(f"{'#'*70}")

        print_graph_structure(graph_id)
        compare_old_vs_new(graph_id)

    print("\n\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
