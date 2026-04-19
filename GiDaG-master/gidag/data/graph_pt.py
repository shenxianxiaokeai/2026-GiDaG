from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from gidag.types import GraphData


def _as_tensor(x, dtype=None) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.tensor(x)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def _concat_features(bundle: Dict, num_nodes: int) -> Optional[torch.Tensor]:
    parts = []
    for key in ["X_categorical", "X_poisson", "X_gaussian", "X", "features"]:
        value = bundle.get(key, None)
        if value is None:
            continue
        t = _as_tensor(value)
        if t.ndim == 2 and t.size(0) == num_nodes and t.size(1) > 0:
            parts.append(t.float())
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
    n = int(adj.size(0))
    edges = []
    for i in range(n):
        neighbors = torch.where(adj[i] > 0)[0]
        for j in neighbors.tolist():
            if i < int(j):
                edges.append((int(i), int(j)))
    return edges


def _to_bidirectional_edge_index(edges_undirected: List[Tuple[int, int]]) -> torch.Tensor:
    if not edges_undirected:
        return torch.empty((2, 0), dtype=torch.long)
    forward = torch.tensor(edges_undirected, dtype=torch.long)
    backward = torch.stack([forward[:, 1], forward[:, 0]], dim=1)
    all_edges = torch.cat([forward, backward], dim=0)
    return all_edges.t().contiguous()


def _infer_dataset_name(dataset_path: Path, override_name: Optional[str]) -> str:
    if override_name is not None and len(override_name.strip()) > 0:
        return override_name.strip()
    if dataset_path.parent.name and dataset_path.parent.name.lower() != "data":
        return dataset_path.parent.name
    return dataset_path.stem


def load_graph_pt(dataset_path: str, dataset_name: Optional[str] = None) -> GraphData:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError("Dataset not found: {p}".format(p=p))

    # Use weights_only=False for compatibility with older .pt files containing numpy arrays.
    bundle = torch.load(str(p), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError("Expected dict payload in {p}".format(p=p))

    adj = bundle.get("A", None)
    if adj is None:
        adj = bundle.get("adj", None)
    if adj is None:
        raise KeyError("Missing adjacency key 'A' in dataset: {p}".format(p=p))
    adj = _as_tensor(adj, dtype=torch.float32)

    if adj.ndim == 3:
        num_nodes = int(adj.size(1))
    elif adj.ndim == 2:
        num_nodes = int(adj.size(0))
    else:
        raise ValueError("Unsupported adjacency shape: {s}".format(s=tuple(adj.shape)))

    labels = bundle.get("y_true", None)
    if labels is None:
        labels = bundle.get("y", None)
    if labels is None:
        labels = bundle.get("labels", None)
    if labels is None:
        raise KeyError("Missing labels key (y_true/y/labels) in dataset: {p}".format(p=p))
    labels = _as_tensor(labels, dtype=torch.long)
    labels = _reindex_labels(labels)

    edges_undirected = _extract_edges_from_adj(adj)
    edge_index = _to_bidirectional_edge_index(edges_undirected)
    features = _concat_features(bundle, num_nodes=num_nodes)

    return GraphData(
        name=_infer_dataset_name(p, dataset_name),
        num_nodes=num_nodes,
        num_classes=int(torch.unique(labels).numel()),
        labels=labels,
        features=features,
        edge_index=edge_index,
        edges_undirected=edges_undirected,
    )
