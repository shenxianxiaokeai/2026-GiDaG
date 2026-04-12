from dataclasses import dataclass, field
from typing import List


@dataclass
class GEEConfig:
    replicates: int = 8
    num_iter: int = 80
    kmeans_max_iter: int = 300


@dataclass
class GNNConfig:
    num_epochs: int = 4000
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    eval_interval: int = 20
    loss_patience: int = 80
    hidden_dim: int = 128
    dropout: float = 0.2
    cluster_head: str = "kmeans"  # kmeans | argmax


@dataclass
class ExperimentConfig:
    dataset_path: str = r"E:\Article\2026 GiDaG\GiDaG-master\data\EmailEU\EmailEU.pt"
    output_dir: str = "outputs/email_eu_cuda"
    methods: List[str] = field(default_factory=lambda: ["GEE", "GNN", "GiDaG", "GiDaG-C"])
    num_runs: int = 10
    seed_start: int = 901
    device: str = "cuda"
    save_run_npz: bool = True
    save_aggregate_npz: bool = True
    gee: GEEConfig = field(default_factory=GEEConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
