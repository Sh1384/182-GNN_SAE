# Virtual Graph Generation and Analysis

## Motif-Driven Graph Construction
The `GraphMotifGenerator` class instantiates four canonical directed motifs—feedforward loop, feedback loop, single-input module, and cascade—while keeping edge weights continuous in the range [0, 1] so downstream GNNs can reason about varying interaction strengths. Each motif builder samples disjoint node identifiers, lays down the motif-specific edge pattern, and annotates nodes with human-readable labels for traceability (`virtual_graphs/graph_motif_generator.py:31-128`).

### Single-Motif Graphs
To create single-motif graphs, the generator expands each motif to a target of three to four nodes (padding with isolated nodes if necessary), relabels nodes sequentially, and stamps the motif type into the graph metadata. This routine is repeated 1,000 times per motif, yielding 4,000 balanced graphs that are serialized as pickles under `virtual_graphs/data/single_motif_graphs` for rapid loading during training (`virtual_graphs/graph_motif_generator.py:200-243`).

### Mixed-Motif Graphs
For mixed motifs, the generator samples two or three motif templates, offsets their node indices before composing them into a single graph, and inserts extra random inter-motif edges with probability between 0.2 and 0.3 to ensure rich connectivity. The merged graph is then relabeled sequentially and written to disk, with 1,000 such graphs providing a heterogeneous test bed for multi-motif reasoning tasks (`virtual_graphs/graph_motif_generator.py:245-346`).

## Expression Simulation Pipeline
Beyond topology, each stored graph supports synthetic expression dynamics via an iterative update rule: \(x_{t+1} = (1-\gamma)x_t + \gamma\cdot\sigma(Wx_t) + \epsilon\). The simulator initializes random expression states, repeatedly applies graph-weighted sigmoid transformations with Gaussian noise, clips results to [0, 1], and optionally records full trajectories before persisting adjacency matrices, node labels, and expression traces to disk for later supervision (`virtual_graphs/graph_motif_generator.py:348-504`).

## Analysis Workflow
The `graph_data_analysis.ipynb` notebook reloads every pickled graph, leveraging helper routines such as `load_graphs_from_directory`, `extract_graph_properties`, and a notebook-specific `simulate_expression` to recompute key metrics (`virtual_graphs/graph_data_analysis.ipynb:62-154`). The pipeline first tallies how many graphs exist per motif category to confirm dataset balance, then iterates over graphs to compute node/edge counts, density, degree distributions, and edge-weight statistics for downstream visualization.

## Dataset Highlights
The analysis confirms that the corpus contains 4,000 single-motif graphs (1,000 of each motif) plus 1,000 mixed-motif graphs, for a total of 5,000 labeled instances. These counts are echoed in both the loading log and overview summary tables within the notebook, validating that the generation scripts populated every expected directory (`virtual_graphs/graph_data_analysis.ipynb:51-55,102`).

## Structural Signatures
Sampling 200 graphs per motif reveals distinct degree signatures: feedforward loops show the highest mean in-degree and out-degree (≈0.85) thanks to their triangular wiring, cascades average 0.75 degrees reflecting their chain topology, feedback loops remain tightly centered around reciprocal pairs, and single-input modules display the widest out-degree spread (std ≈1.49, max 4) because the regulator can broadcast to four targets. These statistics were derived from violin plots and aggregate tables computed in the notebook, corroborating that the synthetic motifs preserve their intended structural identities (`virtual_graphs/graph_data_analysis.ipynb:1742-1779`).
