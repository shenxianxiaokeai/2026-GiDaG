# Algorithm Notes

## Core Pipeline

1. Build structural embedding with **Unsupervised GEE**.
2. Use GEE embedding as initialization for GiDaG branch.
3. Train residual GCN with **DMoN** objective.
4. Obtain cluster labels by `kmeans` (default) or `argmax`.
5. Build final `GiDaG` by concatenating `GEE` and `GiDaG-backbone` embeddings (old name: `GiDaG-C`).

## GNN Backbones

- `paper` backbone
  - Output embedding dimension equals `num_clusters`.
  - DMoN uses `softmax(embedding)` directly.
- `decoupled` backbone
  - Output embedding dimension equals `hidden_dim`.
  - DMoN uses a separate cluster projection head.

## Input Strategy

- `paper` mode: random input for GNN baseline (strict reproducibility).
- `adapted/enhanced` modes:
  - use node attributes if available;
  - otherwise use structural fallback features:
    - `log-degree`
    - `normalized-degree`
    - `normalized-neighbor-average-degree`

## GiDaG Fusion

- `paper/adapted`: fixed fusion (`beta = 1.0`).
- `enhanced`: search `beta` in `[0.5, 0.75, 1.0, 1.25, 1.5]` by unsupervised modularity `Q`.

## Structured Outputs (GiDaG V2)

Per run / per method:

- `relation_matrix`
- `community_soft`
- `edge_disharmony`
- `node_disharmony`
- `bridge_node_score`
- `edge_bridge_score`
- top-k bridge nodes/edges and scores

## Metrics

- `NMI`, `ACC`, `ARI`
- `Q` (graph modularity)

## Disagreement Score (GEE Restart Selection)

- `score = mean(r_c) + 2 * std(r_c)`
- `r_c = (intra_c / extra_c) * (|c| / n)`

Lower is better.
