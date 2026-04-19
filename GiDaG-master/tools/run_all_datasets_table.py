import csv
import json
import time
from pathlib import Path
from typing import Dict, List
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gidag.config import ExperimentConfig  # noqa: E402
from gidag.data import load_graph_pt  # noqa: E402
from gidag.engine import run_email_eu_experiment  # noqa: E402


def _discover_datasets(data_root: Path) -> List[Path]:
    paths = []
    for p in sorted(data_root.rglob("*.pt")):
        # Skip legacy raw source folder; use organized data layout.
        if "data\\data\\" in str(p).replace("/", "\\"):
            continue
        paths.append(p)
    return paths


def _dataset_id_from_path(p: Path, data_root: Path) -> str:
    rel = p.relative_to(data_root)
    return str(rel).replace("\\", "__").replace("/", "__").replace(".pt", "")


def _configure_mode(cfg: ExperimentConfig, mode: str) -> None:
    if mode == "paper":
        cfg.gnn.use_feature_input = False
        cfg.gnn.backbone_mode = "paper"
        cfg.gnn.gidag_c_auto_balance = False
    elif mode == "adapted":
        cfg.gnn.use_feature_input = True
        cfg.gnn.backbone_mode = "paper"
        cfg.gnn.gidag_c_auto_balance = False
    else:
        cfg.gnn.use_feature_input = True
        cfg.gnn.backbone_mode = "decoupled"
        cfg.gnn.gidag_c_auto_balance = True


def _write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_wide_table(status_rows: List[Dict], metric_rows: List[Dict], methods: List[str]) -> List[Dict]:
    metric_map: Dict[str, Dict[str, Dict]] = {}
    for row in metric_rows:
        did = row["dataset_id"]
        m = row["method"]
        metric_map.setdefault(did, {})[m] = row

    wide_rows = []
    for s in status_rows:
        did = s["dataset_id"]
        out = dict(s)
        for m in methods:
            mm = metric_map.get(did, {}).get(m, {})
            out["{m}_NMI".format(m=m)] = mm.get("NMI")
            out["{m}_ACC".format(m=m)] = mm.get("ACC")
            out["{m}_ARI".format(m=m)] = mm.get("ARI")
            out["{m}_Q".format(m=m)] = mm.get("Q")
            out["{m}_time_sec".format(m=m)] = mm.get("time_sec")
        wide_rows.append(out)
    return wide_rows


def main() -> None:
    repo_root = REPO_ROOT
    data_root = repo_root / "data"
    out_root = repo_root / "outputs" / "all_datasets_table_runs"
    table_root = repo_root / "outputs" / "all_datasets_table"
    table_root.mkdir(parents=True, exist_ok=True)

    datasets = _discover_datasets(data_root)
    print("Found {n} organized datasets.".format(n=len(datasets)))

    status_rows: List[Dict] = []
    metric_rows: List[Dict] = []

    mode = "paper"
    num_runs = 1
    base_methods = ["GEE", "GNN", "GiDaG"]

    for idx, ds in enumerate(datasets, start=1):
        did = _dataset_id_from_path(ds, data_root)
        print("\n[{i}/{n}] {d}".format(i=idx, n=len(datasets), d=did))

        status = {
            "dataset_id": did,
            "dataset_path": str(ds),
            "status": "ok",
            "note": "",
            "num_nodes": None,
            "num_classes": None,
            "methods_run": "",
            "elapsed_total_sec": None,
        }
        t0 = time.time()
        try:
            graph = load_graph_pt(str(ds))
            status["num_nodes"] = int(graph.num_nodes)
            status["num_classes"] = int(graph.num_classes)

            methods = list(base_methods)
            # Resource-safe strategy for very large graphs.
            if graph.num_nodes > 8000:
                methods = ["GEE"]
                status["note"] = "num_nodes>8000, skipped GNN/GiDaG for memory/time safety"
            elif graph.num_nodes > 5000:
                status["note"] = "num_nodes>5000, using reduced hyper-params"

            cfg = ExperimentConfig()
            cfg.dataset_path = str(ds)
            cfg.output_dir = str(out_root / did)
            cfg.methods = methods
            cfg.num_runs = num_runs
            cfg.seed_start = 901
            cfg.device = "cuda"
            cfg.save_run_npz = True
            cfg.save_aggregate_npz = True
            cfg.save_structured_outputs = True
            cfg.structured_topk = 20
            cfg.relation_split = 0.5
            cfg.relation_tau = 0.1
            cfg.bridge_alpha = 0.6

            # Quick-but-stable global settings for full-dataset sweep.
            cfg.gee.replicates = 2
            cfg.gee.num_iter = 20
            cfg.gee.kmeans_max_iter = 200
            cfg.gnn.num_epochs = 300
            cfg.gnn.learning_rate = 0.01
            cfg.gnn.weight_decay = 1e-4
            cfg.gnn.eval_interval = 20
            cfg.gnn.loss_patience = 30
            cfg.gnn.hidden_dim = 128
            cfg.gnn.dropout = 0.2
            cfg.gnn.cluster_head = "kmeans"

            if graph.num_nodes > 5000:
                cfg.gee.replicates = 1
                cfg.gee.num_iter = 10
                cfg.gnn.num_epochs = 100
                cfg.gnn.loss_patience = 10

            _configure_mode(cfg, mode=mode)
            result = run_email_eu_experiment(cfg)

            status["methods_run"] = ",".join(methods)
            for srow in result["summary"]:
                metric_rows.append(
                    {
                        "dataset_id": did,
                        "dataset_name": srow["dataset"],
                        "method": srow["method"],
                        "NMI": srow["NMI_mean"],
                        "ACC": srow["ACC_mean"],
                        "ARI": srow["ARI_mean"],
                        "Q": srow["Q_mean"],
                        "time_sec": srow["time_mean_sec"],
                    }
                )

        except Exception as e:
            status["status"] = "failed"
            status["note"] = "{t}: {m}".format(t=type(e).__name__, m=str(e))
            status["methods_run"] = ""
        finally:
            status["elapsed_total_sec"] = float(time.time() - t0)
            status_rows.append(status)

            _write_csv(table_root / "status_partial.csv", status_rows)
            _write_csv(table_root / "metrics_partial.csv", metric_rows)

    wide_rows = _build_wide_table(status_rows, metric_rows, methods=base_methods)

    _write_csv(table_root / "all_datasets_status.csv", status_rows)
    _write_csv(table_root / "all_datasets_metrics_long.csv", metric_rows)
    _write_csv(table_root / "all_datasets_metrics_wide.csv", wide_rows)

    payload = {
        "num_datasets": len(status_rows),
        "num_success": int(sum(1 for r in status_rows if r["status"] == "ok")),
        "num_failed": int(sum(1 for r in status_rows if r["status"] != "ok")),
        "mode": mode,
        "num_runs": num_runs,
        "base_methods": base_methods,
        "status_csv": str(table_root / "all_datasets_status.csv"),
        "metrics_long_csv": str(table_root / "all_datasets_metrics_long.csv"),
        "metrics_wide_csv": str(table_root / "all_datasets_metrics_wide.csv"),
    }
    (table_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
