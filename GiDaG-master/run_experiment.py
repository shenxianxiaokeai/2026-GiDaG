import argparse
import os
import warnings

from gidag.config import ExperimentConfig
from gidag.engine import run_email_eu_experiment

# 可选：在部分 Windows 环境下减少 joblib 的噪声警告输出。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores for the following reason:",
    category=UserWarning,
)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="在 EmailEU 上运行工程化 GiDaG 实验（支持 CUDA/CPU）。"
    )

    # ------------------------------------------------------------------
    # 输入输出与实验范围参数
    # ------------------------------------------------------------------
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=r"E:\Article\2026 GiDaG\GiDaG-master\data\EmailEU\EmailEU.pt",
        help="输入数据集 .pt 文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/email_eu_cuda",
        help="输出目录（保存指标、配置和可视化工件）。",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["GEE", "GNN", "GiDaG", "GiDaG-C"],
        help="要运行的方法列表。常用：GEE GNN GiDaG GiDaG-C",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="重复实验次数（不同随机种子），用于统计稳定性。",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=901,
        help="起始随机种子。第 i 次运行使用 seed = seed_start + i。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备。建议在 Python310CUDA128 环境中使用 cuda。",
    )
    parser.add_argument(
        "--no-run-npz",
        action="store_true",
        help="关闭每次 run 的 NPZ 文件保存。",
    )
    parser.add_argument(
        "--no-aggregate-npz",
        action="store_true",
        help="关闭全量聚合 NPZ 文件保存。",
    )

    # ------------------------------------------------------------------
    # GEE 超参数
    # ------------------------------------------------------------------
    parser.add_argument(
        "--gee-replicates",
        type=int,
        default=8,
        help="GEE 随机重启次数。越大越稳定，但更耗时。",
    )
    parser.add_argument(
        "--gee-num-iter",
        type=int,
        default=80,
        help="每次 GEE 重启的最大迭代轮数（EM-like）。",
    )
    parser.add_argument(
        "--gee-kmeans-max-iter",
        type=int,
        default=300,
        help="GEE 内部 KMeans 的最大迭代次数。",
    )

    # ------------------------------------------------------------------
    # GNN / DMoN 超参数
    # ------------------------------------------------------------------
    parser.add_argument(
        "--gnn-num-epochs",
        type=int,
        default=4000,
        help="GNN 主干训练最大 epoch 数。",
    )
    parser.add_argument(
        "--gnn-lr",
        type=float,
        default=0.01,
        help="Adam 学习率。",
    )
    parser.add_argument(
        "--gnn-weight-decay",
        type=float,
        default=1e-4,
        help="Adam 的 L2 正则系数（weight decay）。",
    )
    parser.add_argument(
        "--gnn-eval-interval",
        type=int,
        default=20,
        help="每 N 个 epoch 检查一次损失是否改善。",
    )
    parser.add_argument(
        "--gnn-loss-patience",
        type=int,
        default=80,
        help="连续若干次检查无改善后触发早停。",
    )
    parser.add_argument(
        "--gnn-hidden-dim",
        type=int,
        default=128,
        help="残差 GCN 主干的隐藏维度。",
    )
    parser.add_argument(
        "--gnn-dropout",
        type=float,
        default=0.2,
        help="GNN 编码器的 dropout 比例。",
    )
    parser.add_argument(
        "--gnn-cluster-head",
        type=str,
        default="kmeans",
        choices=["kmeans", "argmax"],
        help="将 embedding 映射为聚类标签的方式。",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig()
    cfg.dataset_path = args.dataset_path
    cfg.output_dir = args.output_dir
    cfg.methods = args.methods
    cfg.num_runs = args.num_runs
    cfg.seed_start = args.seed_start
    cfg.device = args.device
    cfg.save_run_npz = not args.no_run_npz
    cfg.save_aggregate_npz = not args.no_aggregate_npz

    cfg.gee.replicates = args.gee_replicates
    cfg.gee.num_iter = args.gee_num_iter
    cfg.gee.kmeans_max_iter = args.gee_kmeans_max_iter

    cfg.gnn.num_epochs = args.gnn_num_epochs
    cfg.gnn.learning_rate = args.gnn_lr
    cfg.gnn.weight_decay = args.gnn_weight_decay
    cfg.gnn.eval_interval = args.gnn_eval_interval
    cfg.gnn.loss_patience = args.gnn_loss_patience
    cfg.gnn.hidden_dim = args.gnn_hidden_dim
    cfg.gnn.dropout = args.gnn_dropout
    cfg.gnn.cluster_head = args.gnn_cluster_head
    return cfg


def main() -> None:
    cfg = parse_args()
    run_email_eu_experiment(cfg)


if __name__ == "__main__":
    main()
