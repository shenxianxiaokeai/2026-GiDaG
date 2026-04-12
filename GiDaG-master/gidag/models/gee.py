from typing import List, Tuple

import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from gidag.metrics.disagreement import disagreement_score


class UnsupervisedGEE(object):
    def __init__(
        self,
        edges_weighted: List[Tuple[int, int, float]],
        num_nodes: int,
        num_clusters: int,
        replicates: int = 8,
        num_iter: int = 80,
        kmeans_max_iter: int = 300,
        random_state: int = 0,
    ):
        self.edges = edges_weighted
        self.num_nodes = int(num_nodes)
        self.num_clusters = int(num_clusters)
        self.replicates = int(replicates)
        self.num_iter = int(num_iter)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.rng = np.random.default_rng(random_state)

    def _embed_given_labels(self, labels_2d: np.ndarray) -> np.ndarray:
        k = self.num_clusters
        nk = np.zeros((1, k), dtype=np.float64)
        w = np.zeros((self.num_nodes, k), dtype=np.float64)

        for cluster_id in range(k):
            nk[0, cluster_id] = np.count_nonzero(labels_2d[:, 0] == cluster_id)
        nk[nk == 0] = 1e-10

        for i in range(labels_2d.shape[0]):
            c = int(labels_2d[i, 0])
            if c >= 0:
                w[i, c] = 1.0 / nk[0, c]

        z = np.zeros((self.num_nodes, k), dtype=np.float64)
        for u, v, ew in self.edges:
            u = int(u)
            v = int(v)
            ew = float(ew)
            cu = int(labels_2d[u, 0])
            cv = int(labels_2d[v, 0])

            if cv >= 0:
                z[u, cv] += w[v, cv] * ew
            if cu >= 0 and u != v:
                z[v, cu] += w[u, cu] * ew

        norm = LA.norm(z, axis=1, keepdims=True)
        norm[norm == 0] = 1e-10
        z = np.nan_to_num(z / norm)
        return z

    def fit(self):
        best_score = None
        best_z = None
        best_y = None

        for _ in range(self.replicates):
            y_temp = self.rng.integers(self.num_clusters, size=(self.num_nodes, 1))
            dist_to_centers = None
            labels = None

            for _ in range(self.num_iter):
                z = self._embed_given_labels(y_temp)
                km_seed = int(self.rng.integers(0, np.iinfo(np.int32).max))
                kmeans = KMeans(
                    n_clusters=self.num_clusters,
                    n_init=10,
                    max_iter=self.kmeans_max_iter,
                    random_state=km_seed,
                ).fit(z)
                labels = kmeans.labels_
                dist_to_centers = kmeans.transform(z)

                ari = adjusted_rand_score(y_temp.reshape(-1), labels)
                y_temp = labels.reshape(-1, 1)
                if ari == 1:
                    break

            score = disagreement_score(
                dist_to_centers=dist_to_centers,
                labels=labels,
                num_nodes=self.num_nodes,
                num_clusters=self.num_clusters,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_z = z
                best_y = labels

        return {
            "embeddings": best_z.astype(np.float32),
            "pred_labels": best_y.astype(np.int64),
            "disagreement_score": float(best_score if best_score is not None else 0.0),
        }
