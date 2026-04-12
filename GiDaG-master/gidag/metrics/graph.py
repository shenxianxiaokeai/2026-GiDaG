from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from gidag.metrics.clustering import clustering_metrics


def modularity_q(num_nodes: int, edges_undirected: List[Tuple[int, int]], y_pred: np.ndarray) -> float:
    if len(edges_undirected) == 0:
        return 0.0
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges_undirected)

    communities = []
    y_pred = np.asarray(y_pred, dtype=np.int64)
    for c in np.unique(y_pred):
        members = np.where(y_pred == c)[0]
        if members.size > 0:
            communities.append(set(int(v) for v in members.tolist()))
    if len(communities) <= 1:
        return 0.0
    return float(nx.algorithms.community.quality.modularity(g, communities))


def all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_nodes: int,
    edges_undirected: List[Tuple[int, int]],
) -> Dict[str, float]:
    m = clustering_metrics(y_true, y_pred)
    m["Q"] = modularity_q(num_nodes, edges_undirected, y_pred)
    return m
