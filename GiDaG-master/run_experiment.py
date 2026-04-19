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
        description="运行工程化 GiDaG 实验（支持 CUDA/CPU，含论文复现/增强两种路线）。"
    )

    # ------------------------------------------------------------------
    # 实验模式
    # ------------------------------------------------------------------
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="paper",
        choices=["paper", "adapted", "enhanced"],
        help=(
            "实验路线选择：paper=论文复现优先；"
            "adapted=输入适配（特征/结构特征）但保持论文主干；"
            "enhanced=在 adapted 基础上启用 decoupled GNN 与 GiDaG-C 自动融合权重。"
        ),
    )

    # ------------------------------------------------------------------
    # 输入输出与范围
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
        help="输出目录（指标、工件、结构化输出）。",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["GEE", "GNN", "GiDaG"],
        help="要运行的方法列表。常用：GEE GNN GiDaG（旧名 GiDaG-C 已并入 GiDaG）。",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="重复运行次数（多随机种子）。",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=901,
        help="起始随机种子。第 i 次 run 使用 seed=seed_start+i。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备。建议在 Python310CUDA128 环境下使用 cuda。",
    )
    parser.add_argument(
        "--no-run-npz",
        action="store_true",
        help="关闭每次 run 的 NPZ 保存。",
    )
    parser.add_argument(
        "--no-aggregate-npz",
        action="store_true",
        help="关闭 all-runs 聚合 NPZ 保存。",
    )

    # ------------------------------------------------------------------
    # 结构化输出（GiDaG V2）
    # ------------------------------------------------------------------
    parser.add_argument(
        "--no-structured-outputs",
        action="store_true",
        help="关闭结构化输出（关系矩阵、软分配、失协、桥接分数）。",
    )
    parser.add_argument(
        "--structured-topk",
        type=int,
        default=20,
        help="每个 run 输出 Top-K 桥接节点/边。",
    )
    parser.add_argument(
        "--relation-split",
        type=float,
        default=0.5,
        help="关系矩阵 Sigmoid 的分割阈值 s。",
    )
    parser.add_argument(
        "--relation-tau",
        type=float,
        default=0.1,
        help="关系矩阵 Sigmoid 的温度 tau。",
    )
    parser.add_argument(
        "--bridge-alpha",
        type=float,
        default=0.6,
        help="桥接节点分数融合系数 alpha。",
    )

    # ------------------------------------------------------------------
    # GEE 超参数
    # ------------------------------------------------------------------
    parser.add_argument("--gee-replicates", type=int, default=8, help="GEE 随机重启次数。")
    parser.add_argument("--gee-num-iter", type=int, default=80, help="每次 GEE 的最大迭代轮数。")
    parser.add_argument("--gee-kmeans-max-iter", type=int, default=300, help="GEE 内部 KMeans 最大迭代。")

    # ------------------------------------------------------------------
    # GNN / DMoN 超参数
    # ------------------------------------------------------------------
    parser.add_argument("--gnn-num-epochs", type=int, default=4000, help="GNN 最大训练 epoch。")
    parser.add_argument("--gnn-lr", type=float, default=0.01, help="Adam 学习率。")
    parser.add_argument("--gnn-weight-decay", type=float, default=1e-4, help="Adam weight decay。")
    parser.add_argument("--gnn-eval-interval", type=int, default=20, help="每 N 个 epoch 检查一次 loss。")
    parser.add_argument("--gnn-loss-patience", type=int, default=80, help="早停耐心值。")
    parser.add_argument("--gnn-hidden-dim", type=int, default=128, help="GNN 隐层维度。")
    parser.add_argument("--gnn-dropout", type=float, default=0.2, help="GNN dropout。")
    parser.add_argument(
        "--gnn-cluster-head",
        type=str,
        default="kmeans",
        choices=["kmeans", "argmax"],
        help="从 embedding 得到标签的方式。",
    )

    # 手动覆盖 experiment-mode
    parser.add_argument(
        "--gnn-random-input",
        action="store_true",
        help="强制 GNN 使用随机输入（覆盖模式默认值）。",
    )
    parser.add_argument(
        "--gnn-use-feature-input",
        action="store_true",
        help="强制 GNN 使用特征输入（覆盖模式默认值）。",
    )
    parser.add_argument(
        "--gnn-backbone-mode",
        type=str,
        choices=["paper", "decoupled"],
        default=None,
        help="手动指定 GNN 主干模式（覆盖模式默认值）。",
    )
    parser.add_argument(
        "--gidag-c-auto-balance",
        action="store_true",
        help="手动开启 GiDaG-C 自动融合权重搜索（覆盖模式默认值）。",
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

    cfg.save_structured_outputs = not args.no_structured_outputs
    cfg.structured_topk = args.structured_topk
    cfg.relation_split = args.relation_split
    cfg.relation_tau = args.relation_tau
    cfg.bridge_alpha = args.bridge_alpha

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

    # 模式默认
    if args.experiment_mode == "paper":
        cfg.gnn.use_feature_input = False
        cfg.gnn.backbone_mode = "paper"
        cfg.gnn.gidag_c_auto_balance = False
    elif args.experiment_mode == "adapted":
        cfg.gnn.use_feature_input = True
        cfg.gnn.backbone_mode = "paper"
        cfg.gnn.gidag_c_auto_balance = False
    else:
        cfg.gnn.use_feature_input = True
        cfg.gnn.backbone_mode = "decoupled"
        cfg.gnn.gidag_c_auto_balance = True

    # 手动覆盖
    if args.gnn_random_input:
        cfg.gnn.use_feature_input = False
    if args.gnn_use_feature_input:
        cfg.gnn.use_feature_input = True
    if args.gnn_backbone_mode is not None:
        cfg.gnn.backbone_mode = args.gnn_backbone_mode
    if args.gidag_c_auto_balance:
        cfg.gnn.gidag_c_auto_balance = True

    return cfg


def main() -> None:
    cfg = parse_args()
    run_email_eu_experiment(cfg)


if __name__ == "__main__":
    main()
