# GiDaG 实验重整思路（效果优先版）

## 目标

在不牺牲论文可比性的前提下，形成一套可重复、可解释、可扩展的实验路线：

1. 主结果必须有“论文复现口径”（paper 模式）。
2. 改进必须是“可控增量”，不能直接污染主结果（adapted/enhanced 模式分离）。
3. 报告必须同时给出聚类指标与结构指标（NMI/ACC/ARI/Q）。

## 三层实验路线

### Layer 1: 主实验（论文口径）

- 模式：`--experiment-mode paper`
- 作用：作为论文主表与对比基线的唯一口径。
- 原则：不做会引入争议的输入增强或结构改造。

### Layer 2: 受控改进（输入适配）

- 模式：`--experiment-mode adapted`
- 改动：仅改变 GNN 输入策略（优先真实特征；无特征时用结构回退特征），保持 paper 主干。
- 作用：证明“输入质量”确实影响结果，但不改变主干方法定义。

### Layer 3: 探索增强（可选）

- 模式：`--experiment-mode enhanced`
- 改动：decoupled GNN + GiDaG-C 自动 beta 搜索。
- 作用：用于附录/补充，不直接替代主结果。

## 评估与汇报规范

1. 主文统一报告 `paper` 模式的多 seed 均值/标准差。
2. `adapted/enhanced` 只作为对照和消融，明确标注“非主口径”。
3. 所有模式统一输出：
   - `metrics_detailed.csv`
   - `metrics_summary.csv`
   - `artifacts_npz`
   - `structured_outputs`

## 推荐执行顺序

1. 先跑 `paper` 10 runs，锁定主表。
2. 再跑 `adapted` 10 runs，验证输入适配贡献。
3. 最后跑 `enhanced`，仅保留稳定有效的增强点。

## 当前建议

- 如果你的目标是“结果稳定 + 审稿可解释”，优先使用：
  - `paper` 做主结果；
  - `adapted` 做改进对照。
