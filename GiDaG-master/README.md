# GiDaG (Engineering Implementation)

Engineering-style implementation of **GiDaG** for graph clustering, with:

- **GEE** structural embedding
- **GNN (DMoN)** refinement
- **GiDaG (fused)** branch (`concat(GEE, GiDaG-backbone) -> KMeans`, old name: GiDaG-C)
- **Structured outputs** for interpretability (relation / disharmony / bridge scores)

## Project Structure

```text
GiDaG-master/
├─ data/
│  └─ EmailEU/EmailEU.pt
├─ gidag/
│  ├─ config.py
│  ├─ data/
│  ├─ models/
│  │  ├─ gee.py
│  │  ├─ gnn_dmon.py
│  │  └─ gidag.py
│  ├─ metrics/
│  ├─ engine/
│  │  ├─ experiment.py
│  │  └─ structured_output.py
│  └─ utils/
├─ docs/
│  ├─ ALGORITHM.md
│  └─ EXPERIMENT_STRATEGY_ZH.md
├─ run_experiment.py
└─ outputs/
```

## Environment

```powershell
E:\anaconda3\envs\Python310CUDA128\python.exe --version
```

## Quick Start

```powershell
E:\anaconda3\envs\Python310CUDA128\python.exe run_experiment.py --device cuda --experiment-mode paper --num-runs 10 --seed-start 901 --output-dir outputs/email_eu_paper_10runs
```


## Experiment Modes

- `paper`
  - `GNN` uses random input
  - `GNN backbone = paper`
  - `GiDaG` uses fixed fusion weight (beta=1.0)
- `adapted`
  - `GNN` prefers node features; if missing, uses structural fallback features
  - `GNN backbone = paper`
  - `GiDaG` uses fixed fusion weight (beta=1.0)
- `enhanced`
  - `GNN` uses feature/structural input
  - `GNN backbone = decoupled`
  - `GiDaG` auto-searches fusion weight by unsupervised modularity

## Input

Default dataset path:

- `E:\Article\2026 GiDaG\GiDaG-master\data\EmailEU\EmailEU.pt`

## Output

Under `--output-dir`:

- `config.json`
- `dataset_info.json`
- `metrics_detailed.csv`
- `metrics_summary.csv`
- `metrics_summary.json`
- `artifacts_npz/`
  - `run_XXX_seed_YYY.npz`
  - `all_runs_aggregate.npz`
  - `all_runs_aggregate_index.json`
- `structured_outputs/`
  - `<method>/run_XXX_seed_YYY_<method>.npz`
  - `<method>/run_XXX_seed_YYY_<method>_summary.json`
  - `structured_index.json`

## Metrics

- `NMI`
- `ACC`
- `ARI`
- `Q` (modularity)


