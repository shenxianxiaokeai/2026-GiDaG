import argparse
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch


def _read_pickle(path: Path):
    with path.open("rb") as f:
        # Planetoid 原始文件是 Python2 pickle，需要 latin1 兼容读取。
        return pickle.load(f, encoding="latin1")


def _read_index(path: Path) -> np.ndarray:
    lines = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return np.asarray([int(x) for x in lines], dtype=np.int64)


def _load_planetoid(dataset_key: str, planetoid_dir: Path) -> Tuple[np.ndarray, np.ndarray, sp.csr_matrix]:
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    x, y, tx, ty, allx, ally, graph = [
        _read_pickle(planetoid_dir / f"ind.{dataset_key}.{name}") for name in names
    ]
    test_idx_reorder = _read_index(planetoid_dir / f"ind.{dataset_key}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_key == "citeseer":
        # Citeseer 存在孤立节点，需要按连续区间补齐 test feature/label。
        test_idx_range_full = np.arange(test_idx_range.min(), test_idx_range.max() + 1)
        tx_ext = sp.lil_matrix((len(test_idx_range_full), x.shape[1]), dtype=tx.dtype)
        tx_ext[test_idx_range - test_idx_range.min(), :] = tx
        tx = tx_ext
        ty_ext = np.zeros((len(test_idx_range_full), y.shape[1]), dtype=ty.dtype)
        ty_ext[test_idx_range - test_idx_range.min(), :] = ty
        ty = ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    y_true = np.asarray(labels.argmax(axis=1), dtype=np.int64)

    n = int(labels.shape[0])
    rows: List[int] = []
    cols: List[int] = []
    for src, neighs in graph.items():
        s = int(src)
        for dst in neighs:
            rows.append(s)
            cols.append(int(dst))

    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    adj = adj.maximum(adj.T).tocsr()
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0

    feat = np.asarray(features.todense(), dtype=np.float32)
    return feat, y_true, adj


def _build_bundle(dataset_name: str, feat: np.ndarray, y_true: np.ndarray, adj: sp.csr_matrix, source_dir: Path) -> Dict:
    n, fdim = feat.shape
    num_classes = int(np.unique(y_true).size)
    adj_dense = adj.toarray().astype(np.float32)

    return {
        "A": torch.from_numpy(adj_dense).unsqueeze(0).float(),
        "X": torch.from_numpy(feat).float(),
        "time": torch.zeros((n,), dtype=torch.float32),
        "y_true": torch.from_numpy(y_true).long(),
        "meta": {
            "dataset": dataset_name,
            "source": "Planetoid original features",
            "planetoid_dir": str(source_dir),
            "num_nodes": int(n),
            "num_features": int(fdim),
            "num_classes": int(num_classes),
            "use_features": True,
        },
    }


def _backup_if_needed(path: Path, backup_tag: str) -> None:
    backup = path.with_name(f"{path.stem}.{backup_tag}.pt")
    if backup.exists():
        return
    shutil.copy2(path, backup)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="将 Planetoid 原始 Cora/Citeseer 转换为带原始特征的项目 .pt 数据集。")
    p.add_argument(
        "--planetoid-dir",
        type=str,
        default=r"E:\Article\2026CausalDetection\master\data\resource\planetoid-master\data",
        help="Planetoid 原始文件目录（包含 ind.cora.x 等）。",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="data",
        help="项目数据根目录（会写到 data/Cora/Cora.pt 与 data/Citeseer/Citeseer.pt）。",
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=["Cora", "Citeseer"],
        help="要恢复的数据集。",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖现有目标文件；覆盖前自动备份。",
    )
    p.add_argument(
        "--backup-tag",
        type=str,
        default="structure_only.bak",
        help="覆盖前备份标签。",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    planetoid_dir = Path(args.planetoid_dir)
    output_root = Path(args.output_root)

    name_map = {
        "cora": ("Cora", "cora"),
        "citeseer": ("Citeseer", "citeseer"),
    }

    if not planetoid_dir.exists():
        raise FileNotFoundError(f"Planetoid dir not found: {planetoid_dir}")

    for ds in args.datasets:
        key = ds.lower()
        if key not in name_map:
            raise ValueError(f"Unsupported dataset: {ds}")
        ds_name, raw_key = name_map[key]

        feat, y_true, adj = _load_planetoid(raw_key, planetoid_dir)
        bundle = _build_bundle(ds_name, feat, y_true, adj, planetoid_dir)

        out_dir = output_root / ds_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{ds_name}.pt"

        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists. Add --overwrite to replace it.")
        if out_path.exists() and args.overwrite:
            _backup_if_needed(out_path, args.backup_tag)

        torch.save(bundle, out_path)
        print(
            "[OK] {name}: nodes={n}, feat_dim={f}, classes={c}, saved={path}".format(
                name=ds_name,
                n=bundle["meta"]["num_nodes"],
                f=bundle["meta"]["num_features"],
                c=bundle["meta"]["num_classes"],
                path=out_path,
            )
        )


if __name__ == "__main__":
    main()
