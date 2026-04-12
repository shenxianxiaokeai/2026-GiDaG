from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from gidag.config import GEEConfig, GNNConfig
from gidag.models.gee import UnsupervisedGEE
from gidag.models.gnn_dmon import train_gnn_dmon
from gidag.types import GraphData


class GiDaGPipeline(object):
    def __init__(self, gee_cfg: GEEConfig, gnn_cfg: GNNConfig, device: str = "cuda"):
        self.gee_cfg = gee_cfg
        self.gnn_cfg = gnn_cfg
        self.device = device

    @staticmethod
    def _edge_list(edges_undirected: List[Tuple[int, int]]):
        return [(int(u), int(v), 1.0) for u, v in edges_undirected]

    @staticmethod
    def _random_init(num_nodes: int, num_clusters: int) -> torch.Tensor:
        x = torch.empty((num_nodes, num_clusters), dtype=torch.float32)
        nn.init.xavier_uniform_(x)
        return x

    def run_gee(self, graph: GraphData, seed: int) -> Dict:
        model = UnsupervisedGEE(
            edges_weighted=self._edge_list(graph.edges_undirected),
            num_nodes=graph.num_nodes,
            num_clusters=graph.num_classes,
            replicates=self.gee_cfg.replicates,
            num_iter=self.gee_cfg.num_iter,
            kmeans_max_iter=self.gee_cfg.kmeans_max_iter,
            random_state=seed,
        )
        return model.fit()

    def run_gnn(self, graph: GraphData, x_init: torch.Tensor) -> Dict:
        return train_gnn_dmon(
            x_init=x_init,
            edge_index=graph.edge_index,
            num_clusters=graph.num_classes,
            num_epochs=self.gnn_cfg.num_epochs,
            learning_rate=self.gnn_cfg.learning_rate,
            weight_decay=self.gnn_cfg.weight_decay,
            eval_interval=self.gnn_cfg.eval_interval,
            loss_patience=self.gnn_cfg.loss_patience,
            hidden_dim=self.gnn_cfg.hidden_dim,
            dropout=self.gnn_cfg.dropout,
            cluster_head=self.gnn_cfg.cluster_head,
            device=self.device,
        )

    def run_gnn_random(self, graph: GraphData) -> Dict:
        x_init = self._random_init(graph.num_nodes, graph.num_classes)
        return self.run_gnn(graph, x_init=x_init)

    def run_gidag(self, graph: GraphData, gee_embeddings: np.ndarray) -> Dict:
        x_init = torch.tensor(gee_embeddings, dtype=torch.float32)
        return self.run_gnn(graph, x_init=x_init)

    def run_gidag_c(self, graph: GraphData, gee_embeddings: np.ndarray, gidag_embeddings: np.ndarray) -> Dict:
        fused = np.concatenate([gee_embeddings, gidag_embeddings], axis=1)
        km = KMeans(n_clusters=graph.num_classes, n_init=20, random_state=0).fit(fused)
        return {
            "embeddings": fused.astype(np.float32),
            "pred_labels": km.labels_.astype(np.int64),
        }
