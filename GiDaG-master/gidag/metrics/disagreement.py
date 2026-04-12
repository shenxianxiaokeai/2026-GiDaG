import numpy as np


def disagreement_score(
    dist_to_centers: np.ndarray,
    labels: np.ndarray,
    num_nodes: int,
    num_clusters: int,
) -> float:
    """
    Disagreement score used to choose the best restart in GEE.

    score = mean(r_c) + 2 * std(r_c), where
    r_c = (intra_c / extra_c) * (|c| / n)
    """
    labels = np.asarray(labels, dtype=np.int64)
    d2 = np.asarray(dist_to_centers, dtype=np.float64) ** 2
    label_count = np.bincount(labels, minlength=num_clusters)
    label_count[label_count == 0] = 1

    intra_sum = np.zeros((num_clusters,), dtype=np.float64)
    for i in range(num_nodes):
        c = labels[i]
        intra_sum[c] += d2[i, c]

    extra_sum = (np.sum(d2, axis=0) - intra_sum) ** 0.5
    intra_norm = intra_sum ** 0.5 / label_count
    extra_norm = extra_sum / np.maximum(num_nodes - label_count, 1)

    ratio = intra_norm / np.maximum(extra_norm, 1e-10)
    ratio = ratio * label_count / max(num_nodes, 1)
    return float(np.mean(ratio) + 2.0 * np.std(ratio))
