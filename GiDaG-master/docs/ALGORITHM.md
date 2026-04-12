# Algorithm Notes

## GiDaG Core

1. Build structural embedding with **Unsupervised GEE**.
2. Use GEE embedding as node feature initialization.
3. Train residual GCN with **DMoN loss**.
4. Obtain cluster assignments from embedding (`kmeans` head by default).

## GG-inspired Extension (GiDaG-C)

Following the GG paper idea of combining views, we add a fused branch:

- `GiDaG-C = concat(GEE_embedding, GiDaG_embedding) -> KMeans`

This often improves separation for heterophilic or sparse label structures.

## Metrics

- `NMI`, `ACC`, `ARI`: clustering quality vs. labels.
- `Q`: graph modularity of predicted communities.

## Disagreement Score

For GEE multi-restart selection, we use disagreement score:

- `score = mean(r_c) + 2 * std(r_c)`
- `r_c = (intra_c / extra_c) * (|c| / n)`

Lower score indicates better compactness + separation.

