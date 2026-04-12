# GiDaG (Engineering Implementation)

This repository contains an engineering-style implementation of **GiDaG** for graph clustering, aligned with the innovation ideas in GiDaG/GG:

- **GEE structural embedding** (unsupervised graph encoder embedding)
- **GNN refinement** with **DMoN** objective
- **GG-style fusion branch (GiDaG-C)**: `concat(GEE, GiDaG) -> KMeans`

## Project Structure

```text
GiDaG-master/
├─ data/
│  └─ EmailEU/EmailEU.pt
├─ gidag/
│  ├─ config.py
│  ├─ types.py
│  ├─ data/
│  │  └─ email_eu.py
│  ├─ models/
│  │  ├─ gee.py
│  │  ├─ gnn_dmon.py
│  │  └─ gidag.py
│  ├─ metrics/
│  │  ├─ clustering.py
│  │  ├─ graph.py
│  │  └─ disagreement.py
│  ├─ engine/
│  │  └─ experiment.py
│  └─ utils/
├─ configs/
│  └─ email_eu_cuda_example.json
├─ docs/
│  └─ ALGORITHM.md
├─ run_experiment.py
├─ requirements.txt
└─ outputs/
```

## Environment

Use the requested CUDA environment:

```powershell
E:\anaconda3\envs\Python310CUDA128\python.exe --version
```

## Run (CUDA)

```powershell
E:\anaconda3\envs\Python310CUDA128\python.exe run_experiment.py --device cuda --num-runs 10 --output-dir outputs/email_eu_cuda_10runs
```

## Input

Default dataset path:

- `E:\Article\2026 GiDaG\GiDaG-master\data\EmailEU\EmailEU.pt`

## Output

Under `outputs/email_eu_cuda_10runs/` (or your custom `--output-dir`):

- `config.json` (full experiment configuration)
- `dataset_info.json` (dataset summary / input metadata)
- `metrics_detailed.csv` (all runs, all methods)
- `metrics_summary.csv` (mean/std by method)
- `metrics_summary.json` (JSON summary)
- `artifacts_npz/` (visualization-ready artifacts)
  - `run_XXX_seed_YYY.npz` (per-run labels/predictions/embeddings/edges)
  - `all_runs_aggregate.npz` (stacked arrays across runs)
  - `all_runs_aggregate_index.json` (key mapping and tensor shapes)

Metrics:

- `NMI`
- `ACC`
- `ARI`
- `Q` (modularity)

## Latest 10-run CUDA Result (EmailEU)

- `GEE`: NMI `0.7082`, ACC `0.5264`, ARI `0.4288`, Q `0.2321`
- `GNN`: NMI `0.2060`, ACC `0.1010`, ARI `0.0005`, Q `-0.0009`
- `GiDaG`: NMI `0.6089`, ACC `0.4228`, ARI `0.3665`, Q `0.1952`
- `GiDaG-C`: NMI `0.7077`, ACC `0.5315`, ARI `0.4387`, Q `0.2361`
