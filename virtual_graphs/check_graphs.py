import pickle
from pathlib import Path

data_dir = Path('./data/all_graphs')
graphs_dir = data_dir / 'raw_graphs'

# Check first few graphs and some key indices
test_indices = [0, 1, 2, 50, 100, 500, 999, 1000, 1001, 1500, 2000, 3000, 4000]

print("Checking graph pickle files:")
print("-" * 60)

for i in test_indices:
    graph_path = graphs_dir / f'graph_{i}.pkl'
    if graph_path.exists():
        with open(graph_path, 'rb') as f:
            G = pickle.load(f)

        # Check graph attributes
        motif_info = "UNKNOWN"
        if 'motif_type' in G.graph:
            motif_info = f"Single: {G.graph['motif_type']}"
        elif 'motif_composition' in G.graph:
            motif_info = f"Mixed: {G.graph['motif_composition']}"
        else:
            motif_info = "NO MOTIF INFO IN GRAPH ATTRIBUTES!"

        print(f"Graph {i:4d}: {motif_info:40s} (nodes={len(G.nodes())}, edges={len(G.edges())})")
    else:
        print(f"Graph {i:4d}: FILE NOT FOUND")

print("-" * 60)
