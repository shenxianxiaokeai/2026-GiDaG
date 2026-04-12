from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def clustering_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return 0.0

    dim = int(max(y_true.max(), y_pred.max()) + 1)
    w = np.zeros((dim, dim), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        w[p, t] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return float(w[row, col].sum() / y_true.size)


def clustering_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "NMI": float(normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")),
        "ACC": float(clustering_acc(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
    }
