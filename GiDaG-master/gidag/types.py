from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class GraphData:
    name: str
    num_nodes: int
    num_classes: int
    labels: torch.Tensor
    features: Optional[torch.Tensor]
    edge_index: torch.Tensor  # bidirectional COO for GNN
    edges_undirected: List[Tuple[int, int]]  # unique undirected edges for GEE / Q
