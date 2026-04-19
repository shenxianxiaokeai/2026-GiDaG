from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gidag.utils.io import ensure_dir, write_json


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    denom = expv.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return expv / denom


def build_structured_outputs(
    embeddings: np.ndarray,
    pred_labels: np.ndarray,
    edges_undirected: List[Tuple[int, int]],
    num_nodes: int,
    num_clusters: int,
    relation_split: float = 0.5,
    relation_tau: float = 0.1,
    bridge_alpha: float = 0.6,
    topk: int = 20,
) -> Dict[str, np.ndarray]:
    emb = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(pred_labels, dtype=np.int64)
    n = int(num_nodes)
    k = int(num_clusters)

    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-12
    emb_n = emb / norm
    sim = emb_n @ emb_n.T
    tau = max(float(relation_tau), 1e-6)
    rel = 1.0 / (1.0 + np.exp(-(sim - float(relation_split)) / tau))
    np.fill_diagonal(rel, 1.0)

    d = emb.shape[1]
    centroids = np.zeros((k, d), dtype=np.float32)
    global_mean = emb.mean(axis=0)
    for cid in range(k):
        idx = np.where(y == cid)[0]
        if idx.size > 0:
            centroids[cid] = emb[idx].mean(axis=0)
        else:
            centroids[cid] = global_mean
    dist2 = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    community_soft = _row_softmax(-dist2)

    if edges_undirected:
        edges_arr = np.asarray(edges_undirected, dtype=np.int64)
    else:
        edges_arr = np.empty((0, 2), dtype=np.int64)

    m = edges_arr.shape[0]
    edge_disharmony = np.zeros((m,), dtype=np.float32)
    node_disharmony_sum = np.zeros((n,), dtype=np.float32)
    degree = np.zeros((n,), dtype=np.float32)

    for eid in range(m):
        u = int(edges_arr[eid, 0])
        v = int(edges_arr[eid, 1])
        dval = float(1.0 - rel[u, v])
        edge_disharmony[eid] = dval
        node_disharmony_sum[u] += dval
        node_disharmony_sum[v] += dval
        degree[u] += 1.0
        degree[v] += 1.0

    degree[degree == 0] = 1.0
    node_disharmony = node_disharmony_sum / degree

    boundary_score = 1.0 - community_soft.max(axis=1)
    nd_min = float(node_disharmony.min()) if node_disharmony.size else 0.0
    nd_max = float(node_disharmony.max()) if node_disharmony.size else 1.0
    nd_norm = (node_disharmony - nd_min) / (nd_max - nd_min + 1e-8)
    alpha = float(bridge_alpha)
    alpha = min(max(alpha, 0.0), 1.0)
    bridge_node_score = alpha * nd_norm + (1.0 - alpha) * boundary_score

    if m > 0:
        cross = (y[edges_arr[:, 0]] != y[edges_arr[:, 1]]).astype(np.float32)
        edge_bridge_score = edge_disharmony * (0.5 + 0.5 * cross)
    else:
        edge_bridge_score = np.zeros((0,), dtype=np.float32)

    topk_node = min(int(topk), n)
    topk_edge = min(int(topk), m)
    top_node_idx = np.argsort(-bridge_node_score)[:topk_node].astype(np.int64)
    top_node_score = bridge_node_score[top_node_idx].astype(np.float32)
    top_edge_idx_order = np.argsort(-edge_bridge_score)[:topk_edge].astype(np.int64)
    top_edge_score = edge_bridge_score[top_edge_idx_order].astype(np.float32)
    top_edge_pairs = edges_arr[top_edge_idx_order] if topk_edge > 0 else np.empty((0, 2), dtype=np.int64)

    return {
        "relation_matrix": rel.astype(np.float32),
        "community_soft": community_soft.astype(np.float32),
        "edge_index_undirected": edges_arr.astype(np.int64),
        "edge_disharmony": edge_disharmony.astype(np.float32),
        "node_disharmony": node_disharmony.astype(np.float32),
        "bridge_node_score": bridge_node_score.astype(np.float32),
        "edge_bridge_score": edge_bridge_score.astype(np.float32),
        "top_bridge_node_idx": top_node_idx,
        "top_bridge_node_score": top_node_score,
        "top_bridge_edge_pairs": top_edge_pairs.astype(np.int64),
        "top_bridge_edge_score": top_edge_score,
    }


def save_structured_artifacts(
    base_dir: Path,
    run_id: int,
    seed: int,
    method_name: str,
    payload: Dict[str, np.ndarray],
) -> Dict:
    ensure_dir(base_dir)
    method_key = method_name.lower().replace("-", "_")
    npz_path = base_dir / "run_{rid:03d}_seed_{seed}_{method}.npz".format(
        rid=run_id, seed=seed, method=method_key
    )
    np.savez_compressed(str(npz_path), **payload)

    top_nodes = [
        {"node_id": int(i), "score": float(s)}
        for i, s in zip(payload["top_bridge_node_idx"].tolist(), payload["top_bridge_node_score"].tolist())
    ]
    top_edges = [
        {
            "u": int(pair[0]),
            "v": int(pair[1]),
            "score": float(score),
        }
        for pair, score in zip(payload["top_bridge_edge_pairs"].tolist(), payload["top_bridge_edge_score"].tolist())
    ]
    summary_path = base_dir / "run_{rid:03d}_seed_{seed}_{method}_summary.json".format(
        rid=run_id, seed=seed, method=method_key
    )
    summary = {
        "run": int(run_id),
        "seed": int(seed),
        "method": method_name,
        "npz_file": npz_path.name,
        "summary_file": summary_path.name,
        "top_bridge_nodes": top_nodes,
        "top_bridge_edges": top_edges,
    }
    write_json(summary_path, summary)
    return summary
