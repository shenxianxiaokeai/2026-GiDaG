from pathlib import Path
from typing import Dict, List, Tuple

import torch

from gidag.types import GraphData


def _concat_features(bundle: Dict, num_nodes: int):
    parts = []
    for key in ["X_categorical", "X_poisson", "X_gaussian"]:
        value = bundle.get(key, None)
        if value is None:
            continue
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        if value.ndim == 2 and value.size(0) == num_nodes and value.size(1) > 0:
            parts.append(value.float())
    if not parts:
        return None
    return torch.cat(parts, dim=1).float()


def _reindex_labels(y: torch.Tensor) -> torch.Tensor:
    y = y.long().view(-1)
    unique = torch.unique(y)
    unique_sorted = torch.sort(unique).values.tolist()
    mapping = {old: new for new, old in enumerate(unique_sorted)}
    remapped = torch.tensor([mapping[int(v)] for v in y.tolist()], dtype=torch.long)
    return remapped


def _extract_edges_from_adj(adj: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Return unique undirected edges as (u, v) with u < v.
    """
    if adj.ndim == 3:
        adj = adj[0]
    adj = (adj > 0).to(torch.int)
    n = adj.size(0)
    edges = []
    for i in range(n):
        row = adj[i]
        neighbors = torch.where(row > 0)[0]
        for j in neighbors.tolist():
            if i < j:
                edges.append((int(i), int(j)))
    return edges


def _to_bidirectional_edge_index(edges_undirected: List[Tuple[int, int]]) -> torch.Tensor:
    if not edges_undirected:
        return torch.empty((2, 0), dtype=torch.long)
    forward = torch.tensor(edges_undirected, dtype=torch.long)
    backward = torch.stack([forward[:, 1], forward[:, 0]], dim=1)
    all_edges = torch.cat([forward, backward], dim=0)
    return all_edges.t().contiguous()


def load_email_eu_pt(dataset_path: str) -> GraphData:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError("Dataset not found: {p}".format(p=p))

    bundle = torch.load(str(p), map_location="cpu")
    if not isinstance(bundle, dict):
        raise TypeError("Expected dict payload in {p}".format(p=p))

    adj = bundle["A"]
    if not torch.is_tensor(adj):
        adj = torch.tensor(adj)

    if adj.ndim == 3:
        num_nodes = int(adj.size(1))
    else:
        num_nodes = int(adj.size(0))

    labels = bundle["y_true"]
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    labels = _reindex_labels(labels)

    edges_undirected = _extract_edges_from_adj(adj)
    edge_index = _to_bidirectional_edge_index(edges_undirected)
    features = _concat_features(bundle, num_nodes=num_nodes)

    return GraphData(
        name="EmailEU",
        num_nodes=num_nodes,
        num_classes=int(torch.unique(labels).numel()),
        labels=labels,
        features=features,
        edge_index=edge_index,
        edges_undirected=edges_undirected,
    )
