import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from gidag.config import ExperimentConfig
from gidag.data import load_graph_pt
from gidag.engine.structured_output import build_structured_outputs, save_structured_artifacts
from gidag.metrics.graph import all_metrics
from gidag.models import GiDaGPipeline
from gidag.utils.io import ensure_dir, write_csv, write_json
from gidag.utils.seed import set_seed


def _resolve_device(device: str) -> str:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError("Unsupported device: {d}".format(d=device))


def _method_key(method_name: str) -> str:
    key = []
    for ch in method_name.lower():
        if ch.isalnum():
            key.append(ch)
        else:
            key.append("_")
    return "".join(key).strip("_")


def _summarize(rows: List[Dict]) -> List[Dict]:
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["method"])
        grouped.setdefault(key, []).append(row)

    summary = []
    for (dataset, method), items in grouped.items():
        def stat(name: str):
            values = np.asarray([float(x[name]) for x in items], dtype=np.float64)
            return float(values.mean()), float(values.std(ddof=0))

        nmi_m, nmi_s = stat("NMI")
        acc_m, acc_s = stat("ACC")
        ari_m, ari_s = stat("ARI")
        q_m, q_s = stat("Q")
        t_m, t_s = stat("time_sec")

        row = {
            "dataset": dataset,
            "method": method,
            "runs": len(items),
            "NMI_mean": nmi_m,
            "NMI_std": nmi_s,
            "ACC_mean": acc_m,
            "ACC_std": acc_s,
            "ARI_mean": ari_m,
            "ARI_std": ari_s,
            "Q_mean": q_m,
            "Q_std": q_s,
            "time_mean_sec": t_m,
            "time_std_sec": t_s,
            "selected_beta_mean": float("nan"),
            "selected_beta_std": float("nan"),
        }
        beta_values = np.asarray([float(x.get("selected_beta", np.nan)) for x in items], dtype=np.float64)
        beta_valid = beta_values[np.isfinite(beta_values)]
        if beta_valid.size > 0:
            row["selected_beta_mean"] = float(beta_valid.mean())
            row["selected_beta_std"] = float(beta_valid.std(ddof=0))
        summary.append(row)

    return sorted(summary, key=lambda x: x["method"])


def _print_summary(summary_rows: List[Dict]) -> None:
    print("\n=== Summary ===")
    print("{:<10} {:<10} {:>8} {:>8} {:>8} {:>8} {:>10}".format("dataset", "method", "NMI", "ACC", "ARI", "Q", "time(s)"))
    for row in summary_rows:
        print(
            "{:<10} {:<10} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.2f}".format(
                row["dataset"],
                row["method"],
                row["NMI_mean"],
                row["ACC_mean"],
                row["ARI_mean"],
                row["Q_mean"],
                row["time_mean_sec"],
            )
        )


def run_email_eu_experiment(config: ExperimentConfig) -> Dict:
    device = _resolve_device(config.device)
    output_dir = Path(config.output_dir)
    ensure_dir(output_dir)

    graph = load_graph_pt(config.dataset_path)
    y_true_np = graph.labels.cpu().numpy().astype(np.int64)

    dataset_info = {
        "name": graph.name,
        "num_nodes": graph.num_nodes,
        "num_classes": graph.num_classes,
        "num_edges_undirected": len(graph.edges_undirected),
        "has_features": graph.features is not None,
        "feature_dim": int(graph.features.size(1)) if graph.features is not None else 0,
        "device": device,
    }
    write_json(output_dir / "dataset_info.json", dataset_info)
    write_json(output_dir / "config.json", asdict(config))

    pipeline = GiDaGPipeline(config.gee, config.gnn, device=device)
    methods_raw = [m.lower() for m in config.methods]
    # 兼容旧别名：将 gidag-c 归一为 gidag（当前 GiDaG 即融合模型）。
    methods = []
    for m in methods_raw:
        if m == "gidag-c":
            methods.append("gidag")
        else:
            methods.append(m)
    method_set = set(methods)

    artifacts_dir = output_dir / "artifacts_npz"
    structured_dir = output_dir / "structured_outputs"
    if config.save_run_npz or config.save_aggregate_npz:
        ensure_dir(artifacts_dir)
    if config.save_structured_outputs:
        ensure_dir(structured_dir)

    detailed_rows = []
    aggregate_store = {}
    structured_index = {
        "dataset": graph.name,
        "num_nodes": int(graph.num_nodes),
        "num_classes": int(graph.num_classes),
        "relation_split": float(config.relation_split),
        "relation_tau": float(config.relation_tau),
        "bridge_alpha": float(config.bridge_alpha),
        "structured_topk": int(config.structured_topk),
        "artifacts": [],
    }

    def record_artifact(
        method_name: str,
        run_id: int,
        seed: int,
        y_pred: np.ndarray,
        embeddings: np.ndarray,
        metrics: Dict[str, float],
        elapsed: float,
        run_payload: Dict,
        selected_beta: Optional[float] = None,
    ) -> None:
        key = _method_key(method_name)
        run_payload["{k}_pred_labels".format(k=key)] = y_pred.astype(np.int64)
        run_payload["{k}_embeddings".format(k=key)] = embeddings.astype(np.float32)
        run_payload["{k}_NMI".format(k=key)] = np.asarray([metrics["NMI"]], dtype=np.float32)
        run_payload["{k}_ACC".format(k=key)] = np.asarray([metrics["ACC"]], dtype=np.float32)
        run_payload["{k}_ARI".format(k=key)] = np.asarray([metrics["ARI"]], dtype=np.float32)
        run_payload["{k}_Q".format(k=key)] = np.asarray([metrics["Q"]], dtype=np.float32)
        run_payload["{k}_time_sec".format(k=key)] = np.asarray([elapsed], dtype=np.float32)
        if selected_beta is not None:
            run_payload["{k}_selected_beta".format(k=key)] = np.asarray([selected_beta], dtype=np.float32)

        bucket = aggregate_store.setdefault(
            key,
            {
                "method_name": method_name,
                "run_id": [],
                "seed": [],
                "pred_labels": [],
                "embeddings": [],
                "NMI": [],
                "ACC": [],
                "ARI": [],
                "Q": [],
                "time_sec": [],
                "selected_beta": [],
            },
        )
        bucket["run_id"].append(int(run_id))
        bucket["seed"].append(int(seed))
        bucket["pred_labels"].append(y_pred.astype(np.int64))
        bucket["embeddings"].append(embeddings.astype(np.float32))
        bucket["NMI"].append(float(metrics["NMI"]))
        bucket["ACC"].append(float(metrics["ACC"]))
        bucket["ARI"].append(float(metrics["ARI"]))
        bucket["Q"].append(float(metrics["Q"]))
        bucket["time_sec"].append(float(elapsed))
        if selected_beta is not None:
            bucket["selected_beta"].append(float(selected_beta))

    def record_method_result(
        method_name: str,
        run_id: int,
        seed: int,
        y_pred: np.ndarray,
        embeddings: np.ndarray,
        elapsed: float,
        run_payload: Dict,
        selected_beta: Optional[float] = None,
    ) -> Dict[str, float]:
        metrics = all_metrics(y_true_np, y_pred, graph.num_nodes, graph.edges_undirected)
        record_artifact(
            method_name=method_name,
            run_id=run_id,
            seed=seed,
            y_pred=y_pred,
            embeddings=embeddings,
            metrics=metrics,
            elapsed=elapsed,
            run_payload=run_payload,
            selected_beta=selected_beta,
        )
        row = {
            "dataset": graph.name,
            "run": run_id,
            "seed": seed,
            "method": method_name,
            **metrics,
            "time_sec": float(elapsed),
            "selected_beta": float(selected_beta) if selected_beta is not None else np.nan,
        }
        detailed_rows.append(row)

        if config.save_structured_outputs:
            method_key = _method_key(method_name)
            method_dir = structured_dir / method_key
            payload = build_structured_outputs(
                embeddings=embeddings,
                pred_labels=y_pred,
                edges_undirected=graph.edges_undirected,
                num_nodes=graph.num_nodes,
                num_clusters=graph.num_classes,
                relation_split=config.relation_split,
                relation_tau=config.relation_tau,
                bridge_alpha=config.bridge_alpha,
                topk=config.structured_topk,
            )
            summary = save_structured_artifacts(
                base_dir=method_dir,
                run_id=run_id,
                seed=seed,
                method_name=method_name,
                payload=payload,
            )
            item = {
                "run": int(run_id),
                "seed": int(seed),
                "method": method_name,
                "method_key": method_key,
                "npz_file": "{k}/{f}".format(k=method_key, f=summary["npz_file"]),
                "summary_file": "{k}/{f}".format(k=method_key, f=summary["summary_file"]),
            }
            if selected_beta is not None:
                item["selected_beta"] = float(selected_beta)
            structured_index["artifacts"].append(item)

        return metrics

    for run_id in range(config.num_runs):
        seed = int(config.seed_start + run_id)
        print("\n[Run {r}/{n}] seed={s}".format(r=run_id + 1, n=config.num_runs, s=seed))
        set_seed(seed)

        gee_output = None
        gidag_output = None
        run_payload = {
            "dataset_name": np.asarray([graph.name]),
            "run_id": np.asarray([run_id], dtype=np.int64),
            "seed": np.asarray([seed], dtype=np.int64),
            "y_true": y_true_np.astype(np.int64),
            "edge_index": graph.edge_index.cpu().numpy().astype(np.int64),
            "edges_undirected": np.asarray(graph.edges_undirected, dtype=np.int64),
        }
        if graph.features is not None:
            run_payload["node_features"] = graph.features.cpu().numpy().astype(np.float32)

        if "gee" in method_set or "gidag" in method_set:
            st = time.time()
            gee_output = pipeline.run_gee(graph, seed=seed)
            elapsed = time.time() - st
            if "gee" in method_set:
                record_method_result(
                    method_name="GEE",
                    run_id=run_id,
                    seed=seed,
                    y_pred=gee_output["pred_labels"],
                    embeddings=gee_output["embeddings"],
                    elapsed=elapsed,
                    run_payload=run_payload,
                )

        if "gnn" in method_set:
            st = time.time()
            gnn_output = pipeline.run_gnn_baseline(graph)
            elapsed = time.time() - st
            record_method_result(
                method_name="GNN",
                run_id=run_id,
                seed=seed,
                y_pred=gnn_output["pred_labels"],
                embeddings=gnn_output["embeddings"],
                elapsed=elapsed,
                run_payload=run_payload,
            )

        if "gidag" in method_set:
            if gee_output is None:
                gee_output = pipeline.run_gee(graph, seed=seed)
            if gidag_output is None:
                gidag_output = pipeline.run_gidag(graph, gee_embeddings=gee_output["embeddings"])

            st = time.time()
            gidag_c_output = pipeline.run_gidag_c(
                graph,
                gee_embeddings=gee_output["embeddings"],
                gidag_embeddings=gidag_output["embeddings"],
            )
            elapsed = time.time() - st
            record_method_result(
                method_name="GiDaG",
                run_id=run_id,
                seed=seed,
                y_pred=gidag_c_output["pred_labels"],
                embeddings=gidag_c_output["embeddings"],
                elapsed=elapsed,
                run_payload=run_payload,
                selected_beta=float(gidag_c_output.get("selected_beta", 1.0)),
            )

        run_rows = [r for r in detailed_rows if int(r["run"]) == run_id]
        run_rows = sorted(run_rows, key=lambda x: x["method"])
        for row in run_rows:
            base = "  {m:8s} | NMI={nmi:.4f} ACC={acc:.4f} ARI={ari:.4f} Q={q:.4f}".format(
                m=row["method"],
                nmi=row["NMI"],
                acc=row["ACC"],
                ari=row["ARI"],
                q=row["Q"],
            )
            if np.isfinite(float(row.get("selected_beta", np.nan))):
                base += " beta={b:.2f}".format(b=float(row["selected_beta"]))
            print(base)

        if config.save_run_npz:
            run_npz_path = artifacts_dir / "run_{rid:03d}_seed_{seed}.npz".format(rid=run_id, seed=seed)
            np.savez_compressed(str(run_npz_path), **run_payload)

    summary_rows = _summarize(detailed_rows)
    _print_summary(summary_rows)

    write_csv(output_dir / "metrics_detailed.csv", detailed_rows)
    write_csv(output_dir / "metrics_summary.csv", summary_rows)
    write_json(output_dir / "metrics_summary.json", {"summary": summary_rows})

    if config.save_structured_outputs:
        write_json(structured_dir / "structured_index.json", structured_index)

    if config.save_aggregate_npz and aggregate_store:
        aggregate_payload = {
            "dataset_name": np.asarray([graph.name]),
            "y_true": y_true_np.astype(np.int64),
            "edge_index": graph.edge_index.cpu().numpy().astype(np.int64),
            "edges_undirected": np.asarray(graph.edges_undirected, dtype=np.int64),
        }
        if graph.features is not None:
            aggregate_payload["node_features"] = graph.features.cpu().numpy().astype(np.float32)

        index_json = {"methods": {}}
        for key, bucket in aggregate_store.items():
            method_name = str(bucket["method_name"])
            pred = np.stack(bucket["pred_labels"], axis=0).astype(np.int64)
            emb = np.stack(bucket["embeddings"], axis=0).astype(np.float32)
            nmi = np.asarray(bucket["NMI"], dtype=np.float32)
            acc = np.asarray(bucket["ACC"], dtype=np.float32)
            ari = np.asarray(bucket["ARI"], dtype=np.float32)
            q = np.asarray(bucket["Q"], dtype=np.float32)
            time_sec = np.asarray(bucket["time_sec"], dtype=np.float32)
            run_ids = np.asarray(bucket["run_id"], dtype=np.int64)
            seeds = np.asarray(bucket["seed"], dtype=np.int64)

            aggregate_payload["{k}_pred_labels".format(k=key)] = pred
            aggregate_payload["{k}_embeddings".format(k=key)] = emb
            aggregate_payload["{k}_NMI".format(k=key)] = nmi
            aggregate_payload["{k}_ACC".format(k=key)] = acc
            aggregate_payload["{k}_ARI".format(k=key)] = ari
            aggregate_payload["{k}_Q".format(k=key)] = q
            aggregate_payload["{k}_time_sec".format(k=key)] = time_sec
            aggregate_payload["{k}_run_id".format(k=key)] = run_ids
            aggregate_payload["{k}_seed".format(k=key)] = seeds
            if bucket["selected_beta"]:
                aggregate_payload["{k}_selected_beta".format(k=key)] = np.asarray(
                    bucket["selected_beta"], dtype=np.float32
                )

            index_json["methods"][key] = {
                "method_name": method_name,
                "pred_labels_shape": list(pred.shape),
                "embeddings_shape": list(emb.shape),
            }

        aggregate_path = artifacts_dir / "all_runs_aggregate.npz"
        np.savez_compressed(str(aggregate_path), **aggregate_payload)
        write_json(artifacts_dir / "all_runs_aggregate_index.json", index_json)

    return {
        "dataset_info": dataset_info,
        "detailed": detailed_rows,
        "summary": summary_rows,
    }
