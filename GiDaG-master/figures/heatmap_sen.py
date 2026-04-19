import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# ============================================================
# 全局字体与样式设置
# ============================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5

# ============================================================
# 配色方案 B (Teal + Orange)
# ============================================================
cmap_f1 = LinearSegmentedColormap.from_list(
    'ieee_teal', ['#F0F9E8', '#CCEBC5', '#A8DDB5', '#7BCCC4',
                  '#4EB3D3', '#2B8CBE', '#0868AC', '#084081'], N=256)
cmap_auroc = LinearSegmentedColormap.from_list(
    'ieee_orange', ['#FFF7EC', '#FEE8C8', '#FDD49E', '#FDBB84',
                    '#FC8D59', '#EF6548', '#D7301F', '#990000'], N=256)

# ============================================================
# ✅ 补充：9×9 仿真数值（你可以直接替换成真实数据）
# ============================================================
np.random.seed(42)  # 固定随机种子，保证可复现

def generate_data(peak_gamma=4, peak_brs=4, noise=0.02):
    """生成中心高、四周低的热图数据（符合论文调参结果）"""
    data = np.zeros((9,9))
    for i in range(9):
        for j in range(9):
            dist = np.sqrt((i-peak_gamma)**2 + (j-peak_brs)**2)
            val = 0.88 - dist * 0.035 + np.random.randn()*noise
            val = np.clip(val, 0.70, 0.92)
            data[i,j] = round(val,3)
    return data

# 生成 8 组 9×9 数据
llama_mh_f1    = generate_data(4,4)
llama_mh_auroc = generate_data(4,4)
llama_tr_f1    = generate_data(4,4)
llama_tr_auroc = generate_data(4,4)
qwen_mh_f1     = generate_data(4,4)
qwen_mh_auroc  = generate_data(4,4)
qwen_tr_f1     = generate_data(4,4)
qwen_tr_auroc  = generate_data(4,4)

# ============================================================
# ✅ 补充：坐标轴标签
# ============================================================
labels_gamma = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]
labels_brs   = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]

# ============================================================
# 绘图流程
# ============================================================
data_list = [
    (llama_mh_f1,    'Llama3 · History · F1',     True),
    (llama_mh_auroc, 'Llama3 · History · AUROC',  False),
    (llama_tr_f1,    'Llama3 · Treat · F1',      True),
    (llama_tr_auroc, 'Llama3 · Treat · AUROC',   False),
    (qwen_mh_f1,     'Qwen8B · History · F1',    True),
    (qwen_mh_auroc,  'Qwen8B · History · AUROC', False),
    (qwen_tr_f1,     'Qwen8B · Treat · F1',      True),
    (qwen_tr_auroc,  'Qwen8B · Treat · AUROC',   False),
]

fig, axes = plt.subplots(4, 2, figsize=(3.5, 6.2),
    gridspec_kw={
        'hspace': 0.16,
        'wspace': 0.04,
        'left': 0.08,
        'right': 0.92,
        'top': 0.97,
        'bottom': 0.04
    })

for idx, (data, title, is_f1) in enumerate(data_list):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    cmap = cmap_f1 if is_f1 else cmap_auroc

    vmin, vmax = data.min() - 0.005, data.max() + 0.005
    im = ax.imshow(data, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax, interpolation='nearest')

    # 数值标注
    for i in range(9):
        for j in range(9):
            val = data[i, j]
            norm_val = (val - vmin) / (vmax - vmin)
            txt_color = 'white' if norm_val > 0.65 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=3.1, fontweight='bold', color=txt_color)

    # 最优值五角星
    best = np.unravel_index(np.argmax(data), data.shape)
    ax.plot(best[1], best[0], marker='*', markersize=5.5,
            markeredgecolor='gold', markerfacecolor='gold', markeredgewidth=0.3)

    ax.set_xticks(range(9))
    ax.set_yticks(range(9))

    # Y轴：仅第一列显示
    if col == 0:
        ax.set_yticklabels(labels_gamma, fontweight='bold')
        ax.set_ylabel(r'$\gamma$', fontsize=8.5, labelpad=5, rotation=0, va='center')
    else:
        ax.set_yticklabels([])

    # X轴：仅最后一行显示
    if row == 3:
        ax.set_xticklabels(labels_brs, fontweight='bold')
        ax.set_xlabel(r'$\lambda_{\mathrm{brs}}$', fontsize=8.5, labelpad=0)
    else:
        ax.set_xticklabels([])

    ax.set_title(title, fontsize=6.5, fontweight='bold', pad=2)
    ax.tick_params(axis='both', which='both', length=1, pad=0.8)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, format='%.2f')
    cbar.ax.tick_params(labelsize=4, length=1, pad=0.5)

# 导出
plt.savefig('heatmap_schemeB_4x2.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01)
plt.savefig('heatmap_schemeB_4x2.png', dpi=600, bbox_inches='tight', pad_inches=0.01)
plt.show()