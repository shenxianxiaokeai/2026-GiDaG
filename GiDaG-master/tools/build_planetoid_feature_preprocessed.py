import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
import torch


def _load_dense_adj(bundle: Dict) -> np.ndarray:
    adj = bundle.get("A", None)
    if adj is None:
        adj = bundle.get("adj", None)
    if adj is None:
        raise KeyError("Missing adjacency key 'A'/'adj' in input bundle.")
    if torch.is_tensor(adj):
        a = adj.detach().cpu().numpy()
    else:
        a = np.asarray(adj)
    if a.ndim == 3:
        a = a[0]
    a = (a > 0).astype(np.int64)
    # 强制无向、无自环。
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0)
    return a


def _load_features(bundle: Dict, num_nodes: int) -> np.ndarray:
    for key in ["X", "features", "X_gaussian", "X_poisson", "X_categorical"]:
        value = bundle.get(key, None)
        if value is None:
            continue
        t = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        if t.ndim == 2 and t.shape[0] == num_nodes and t.shape[1] > 0:
            return t.astype(np.float32)
    raise KeyError("No valid 2D node feature matrix found (expected X/features/etc.).")


def _load_labels(bundle: Dict, num_nodes: int) -> np.ndarray:
    for key in ["y_true", "y", "labels"]:
        value = bundle.get(key, None)
        if value is None:
            continue
        y = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        y = y.reshape(-1).astype(np.int64)
        if y.shape[0] == num_nodes:
            return y
    raise KeyError("No valid labels found (expected y_true/y/labels).")


def _largest_connected_component_nodes(g: nx.Graph) -> Set[int]:
    if g.number_of_nodes() == 0:
        return set()
    comps = list(nx.connected_components(g))
    if not comps:
        return set()
    return set(max(comps, key=len))


def _kcore_nodes(g: nx.Graph, k: int) -> Set[int]:
    if g.number_of_nodes() == 0:
        return set()
    kc = nx.k_core(g, k=k)
    return set(kc.nodes())


def _induce_subgraph(
    adj: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    keep_nodes: Set[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.array(sorted(list(keep_nodes)), dtype=np.int64)
    if nodes.size == 0:
        raise ValueError("No nodes left after filtering.")
    sub_adj = adj[np.ix_(nodes, nodes)].astype(np.int64)
    sub_x = x[nodes].astype(np.float32)
    sub_y = y[nodes].astype(np.int64)
    return sub_adj, sub_x, sub_y, nodes


def _aa_completion_mutual_topk(
    g: nx.Graph,
    completion_ratio: float,
    completion_topk: int,
    completion_mutual: bool,
) -> Tuple[nx.Graph, int]:
    g2 = g.copy()
    e0 = g2.number_of_edges()
    target_add = int(round(float(completion_ratio) * max(e0, 1)))
    if target_add <= 0:
        return g2, 0

    # 对所有 non-edge 计算 Adamic-Adar 分数。
    aa_iter = nx.adamic_adar_index(g2)
    aa_rows: List[Tuple[int, int, float]] = []
    for u, v, s in aa_iter:
        if np.isfinite(s) and s > 0:
            uu, vv = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
            aa_rows.append((uu, vv, float(s)))
    if not aa_rows:
        return g2, 0

    # 每个节点仅保留 top-k 候选。
    by_node: Dict[int, List[Tuple[int, float]]] = {}
    for u, v, s in aa_rows:
        by_node.setdefault(u, []).append((v, s))
        by_node.setdefault(v, []).append((u, s))

    topk_dir: Set[Tuple[int, int]] = set()
    for u, cand in by_node.items():
        cand_sorted = sorted(cand, key=lambda x: x[1], reverse=True)[: int(completion_topk)]
        for v, _ in cand_sorted:
            topk_dir.add((u, int(v)))

    pair_score: Dict[Tuple[int, int], float] = {}
    for u, v, s in aa_rows:
        pair = (u, v)
        if completion_mutual:
            if (u, v) not in topk_dir or (v, u) not in topk_dir:
                continue
        else:
            if (u, v) not in topk_dir and (v, u) not in topk_dir:
                continue
        pair_score[pair] = float(s)

    candidates = sorted(pair_score.items(), key=lambda x: x[1], reverse=True)
    added = 0
    for (u, v), _ in candidates:
        if added >= target_add:
            break
        if g2.has_edge(u, v):
            continue
        g2.add_edge(u, v)
        added += 1
    return g2, added


def _reindex_labels(y: np.ndarray) -> np.ndarray:
    uniq = np.unique(y)
    mapping = {int(old): int(new) for new, old in enumerate(sorted(uniq.tolist()))}
    return np.asarray([mapping[int(v)] for v in y.tolist()], dtype=np.int64)


def build_dataset(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    use_lcc: bool,
    use_kcore: bool,
    k_core: int,
    use_structural_completion: bool,
    completion_ratio: float,
    completion_topk: int,
    completion_mutual: bool,
) -> None:
    bundle = torch.load(str(input_path), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise TypeError("Expected dict payload in {p}".format(p=input_path))

    adj = _load_dense_adj(bundle)
    n0 = int(adj.shape[0])
    x = _load_features(bundle, n0)
    y = _load_labels(bundle, n0)

    g = nx.from_numpy_array(adj)
    keep = set(range(n0))
    if use_lcc:
        keep = keep.intersection(_largest_connected_component_nodes(g))
    if use_kcore:
        g_tmp = g.subgraph(sorted(list(keep))).copy()
        keep = keep.intersection(_kcore_nodes(g_tmp, k=int(k_core)))

    sub_adj, sub_x, sub_y, kept_nodes = _induce_subgraph(adj, x, y, keep)
    sub_y = _reindex_labels(sub_y)
    g_sub = nx.from_numpy_array(sub_adj)

    num_added_edges = 0
    if use_structural_completion:
        g_sub, num_added_edges = _aa_completion_mutual_topk(
            g_sub,
            completion_ratio=float(completion_ratio),
            completion_topk=int(completion_topk),
            completion_mutual=bool(completion_mutual),
        )

    final_adj = nx.to_numpy_array(g_sub, dtype=np.float32)
    final_adj = (final_adj > 0).astype(np.float32)
    np.fill_diagonal(final_adj, 0.0)

    out = {
        "A": torch.from_numpy(final_adj).unsqueeze(0).float(),
        "X": torch.from_numpy(sub_x).float(),
        "time": torch.zeros((sub_x.shape[0],), dtype=torch.float32),
        "y_true": torch.from_numpy(sub_y).long(),
        "meta": {
            "dataset": dataset_name,
            "source": "Planetoid original features + graph preprocessing",
            "input_path": str(input_path),
            "num_nodes": int(sub_x.shape[0]),
            "num_features": int(sub_x.shape[1]),
            "num_classes": int(np.unique(sub_y).size),
            "use_features": True,
            "use_lcc": bool(use_lcc),
            "use_kcore": bool(use_kcore),
            "k_core": int(k_core),
            "use_structural_completion": bool(use_structural_completion),
            "completion_method": "aa",
            "completion_topk": int(completion_topk),
            "completion_mutual": bool(completion_mutual),
            "completion_ratio": float(completion_ratio),
            "num_added_edges": int(num_added_edges),
            "num_removed_edges": 0,
            "kept_nodes_from_original": kept_nodes.tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, str(output_path))
    print(
        "[OK] {ds}: nodes={n}, feat_dim={f}, classes={c}, added_edges={a}, saved={p}".format(
            ds=dataset_name,
            n=out["meta"]["num_nodes"],
            f=out["meta"]["num_features"],
            c=out["meta"]["num_classes"],
            a=out["meta"]["num_added_edges"],
            p=output_path,
        )
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建保留原始特征的 Planetoid 预处理数据集（LCC+k-core+AA补边）。")
    p.add_argument("--cora-input", type=str, default="data/Cora/Cora.pt")
    p.add_argument("--citeseer-input", type=str, default="data/Citeseer/Citeseer.pt")
    p.add_argument("--cora-output", type=str, default="data/Cora/Cora_feat_preprocessed.pt")
    p.add_argument("--citeseer-output", type=str, default="data/Citeseer/Citeseer_feat_preprocessed.pt")
    p.add_argument("--k-core", type=int, default=2)
    p.add_argument("--completion-topk", type=int, default=5)
    p.add_argument("--completion-mutual", action="store_true", default=True)
    p.add_argument("--no-completion-mutual", action="store_true")
    p.add_argument("--cora-completion-ratio", type=float, default=0.12)
    p.add_argument("--citeseer-completion-ratio", type=float, default=0.10)
    p.add_argument("--no-lcc", action="store_true")
    p.add_argument("--no-kcore", action="store_true")
    p.add_argument("--no-completion", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    completion_mutual = True
    if args.no_completion_mutual:
        completion_mutual = False
    elif args.completion_mutual:
        completion_mutual = True

    use_lcc = not bool(args.no_lcc)
    use_kcore = not bool(args.no_kcore)
    use_completion = not bool(args.no_completion)

    build_dataset(
        input_path=Path(args.cora_input),
        output_path=Path(args.cora_output),
        dataset_name="Cora",
        use_lcc=use_lcc,
        use_kcore=use_kcore,
        k_core=int(args.k_core),
        use_structural_completion=use_completion,
        completion_ratio=float(args.cora_completion_ratio),
        completion_topk=int(args.completion_topk),
        completion_mutual=completion_mutual,
    )
    build_dataset(
        input_path=Path(args.citeseer_input),
        output_path=Path(args.citeseer_output),
        dataset_name="Citeseer",
        use_lcc=use_lcc,
        use_kcore=use_kcore,
        k_core=int(args.k_core),
        use_structural_completion=use_completion,
        completion_ratio=float(args.citeseer_completion_ratio),
        completion_topk=int(args.completion_topk),
        completion_mutual=completion_mutual,
    )


if __name__ == "__main__":
    main()
