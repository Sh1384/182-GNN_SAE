#!/usr/bin/env python3
"""
Update All Graph Motif Metadata

Re-scans all graphs and regenerates metadata CSVs with actual detected motifs.
Overwrites existing metadata files with structurally-accurate multi-label assignments.

Usage:
    python update_all_metadata.py
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from detect_actual_motifs import detect_all_motifs


def main():
    # Set up paths
    base_dir = Path(__file__).parent
    raw_graphs_dir = base_dir / "data" / "all_graphs" / "raw_graphs"
    metadata_dir = base_dir / "data" / "all_graphs" / "graph_motif_metadata"

    if not raw_graphs_dir.exists():
        print(f"Error: Graph directory not found: {raw_graphs_dir}")
        sys.exit(1)

    if not metadata_dir.exists():
        print(f"Error: Metadata directory not found: {metadata_dir}")
        sys.exit(1)

    # Get all graph files (sorted numerically)
    graph_files = sorted(
        raw_graphs_dir.glob("graph_*.pkl"),
        key=lambda x: int(x.stem.split('_')[1])
    )

    print("="*70)
    print("GRAPH MOTIF METADATA UPDATE")
    print("="*70)
    print(f"Graph directory: {raw_graphs_dir}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"Found {len(graph_files)} graphs to process")
    print()

    # Track changes
    changed_count = 0
    unchanged_count = 0
    error_count = 0

    # Process each graph
    for graph_path in tqdm(graph_files, desc="Updating metadata"):
        graph_id = int(graph_path.stem.split('_')[1])
        metadata_path = metadata_dir / f"graph_{graph_id}_metadata.csv"

        try:
            # Detect actual motifs
            new_metadata = detect_all_motifs(graph_path)

            # Compare with old if exists
            if metadata_path.exists():
                old_metadata = pd.read_csv(metadata_path, index_col=0)
                if not new_metadata.equals(old_metadata):
                    changed_count += 1
                else:
                    unchanged_count += 1
            else:
                changed_count += 1

            # Save new metadata
            new_metadata.to_csv(metadata_path)

        except Exception as e:
            print(f"\nError processing graph {graph_id}: {e}")
            error_count += 1
            continue

    # Print summary
    print()
    print("="*70)
    print("METADATA UPDATE COMPLETE")
    print("="*70)
    print(f"Total graphs processed: {len(graph_files)}")
    print(f"  Changed: {changed_count}")
    print(f"  Unchanged: {unchanged_count}")
    print(f"  Errors: {error_count}")
    print()

    if error_count > 0:
        print("WARNING: Some graphs had errors. Please review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
