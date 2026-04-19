from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

from gidag.config import GEEConfig, GNNConfig
from gidag.metrics.graph import modularity_q
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

    @staticmethod
    def _structural_fallback_features(num_nodes: int, edges_undirected: List[Tuple[int, int]]) -> torch.Tensor:
        degree = np.zeros((num_nodes,), dtype=np.float32)
        neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
        for u, v in edges_undirected:
            u = int(u)
            v = int(v)
            degree[u] += 1.0
            degree[v] += 1.0
            neighbors[u].append(v)
            neighbors[v].append(u)

        neigh_avg_degree = np.zeros((num_nodes,), dtype=np.float32)
        for i in range(num_nodes):
            if neighbors[i]:
                neigh_avg_degree[i] = float(np.mean(degree[neighbors[i]]))

        degree_max = float(np.max(degree)) if degree.size else 1.0
        neigh_max = float(np.max(neigh_avg_degree)) if neigh_avg_degree.size else 1.0
        if degree_max <= 0.0:
            degree_max = 1.0
        if neigh_max <= 0.0:
            neigh_max = 1.0

        feat = np.stack(
            [
                np.log1p(degree),
                degree / degree_max,
                neigh_avg_degree / neigh_max,
            ],
            axis=1,
        ).astype(np.float32)
        return torch.from_numpy(feat)

    @staticmethod
    def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().float().clone()
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        norm[norm == 0] = 1.0
        return x / norm

    @staticmethod
    def _normalize_np_rows(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1e-12
        return arr / norm

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
            backbone_mode=self.gnn_cfg.backbone_mode,
            device=self.device,
        )

    def run_gnn_random(self, graph: GraphData) -> Dict:
        x_init = self._random_init(graph.num_nodes, graph.num_classes)
        return self.run_gnn(graph, x_init=x_init)

    def run_gnn_baseline(self, graph: GraphData) -> Dict:
        if bool(self.gnn_cfg.use_feature_input):
            if graph.features is not None and int(graph.features.size(1)) > 0:
                return self.run_gnn(graph, x_init=self._normalize_rows(graph.features))
            x_struct = self._structural_fallback_features(graph.num_nodes, graph.edges_undirected)
            return self.run_gnn(graph, x_init=self._normalize_rows(x_struct))
        return self.run_gnn_random(graph)

    def run_gidag(self, graph: GraphData, gee_embeddings: np.ndarray) -> Dict:
        x_init = torch.tensor(gee_embeddings, dtype=torch.float32)
        return self.run_gnn(graph, x_init=x_init)

    def run_gidag_c(self, graph: GraphData, gee_embeddings: np.ndarray, gidag_embeddings: np.ndarray) -> Dict:
        gee = self._normalize_np_rows(gee_embeddings)
        gidag = self._normalize_np_rows(gidag_embeddings)

        if bool(self.gnn_cfg.gidag_c_auto_balance):
            # For non-paper modes, optionally search a better fusion weight with unsupervised modularity.
            betas = [0.5, 0.75, 1.0, 1.25, 1.5]
            best = None
            for beta in betas:
                fused = np.concatenate([gee, float(beta) * gidag], axis=1)
                km = KMeans(n_clusters=graph.num_classes, n_init=20, random_state=0).fit(fused)
                pred = km.labels_.astype(np.int64)
                q = modularity_q(graph.num_nodes, graph.edges_undirected, pred)
                score = (float(q), -float(km.inertia_))
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "beta": float(beta),
                        "fused": fused.astype(np.float32),
                        "pred": pred,
                    }
            return {
                "embeddings": best["fused"],
                "pred_labels": best["pred"],
                "selected_beta": float(best["beta"]),
            }

        fused = np.concatenate([gee, gidag], axis=1)
        km = KMeans(n_clusters=graph.num_classes, n_init=20, random_state=0).fit(fused)
        return {
            "embeddings": fused.astype(np.float32),
            "pred_labels": km.labels_.astype(np.int64),
            "selected_beta": 1.0,
        }
