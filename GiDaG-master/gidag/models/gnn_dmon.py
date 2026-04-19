from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.cluster import KMeans
from torch_geometric.nn import GCNConv


class ResidualGCNPaper(nn.Module):
    """
    Paper-fidelity branch:
      - embedding dimension = num_clusters
      - DMoN loss directly uses softmax(embedding)
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, out_dim, bias=False)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        z = h + self.residual(x)
        z = F.normalize(z, p=2, dim=1)
        c_probs = F.softmax(z, dim=1)
        return z, c_probs


class ResidualGCNDecoupled(nn.Module):
    """
    Adapted branch:
      - embedding dimension = hidden_dim
      - separate cluster projection head for DMoN probabilities
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_clusters: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.residual = nn.Linear(in_dim, hidden_dim, bias=False)
        self.cluster_proj = nn.Linear(hidden_dim, num_clusters, bias=True)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        z = h + self.residual(x)
        z = F.normalize(z, p=2, dim=1)
        c_probs = F.softmax(self.cluster_proj(z), dim=1)
        return z, c_probs


def _build_modularity_matrix(edge_index: torch.Tensor, num_nodes: int, device: str):
    values = torch.ones(edge_index.size(1), dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(
        edge_index, values, size=(num_nodes, num_nodes), device=device
    ).to_dense()
    adj = torch.maximum(adj, adj.T)
    degree = adj.sum(dim=1)
    m = adj.sum() / 2.0
    if float(m.item()) <= 0.0:
        b = torch.zeros_like(adj)
    else:
        b = adj - torch.outer(degree, degree) / (2.0 * m)
    return b, m


def _dmon_loss(c_probs: torch.Tensor, modularity_matrix: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    n, k = c_probs.shape
    if float(m.item()) <= 0.0:
        modularity_term = torch.tensor(0.0, device=c_probs.device)
    else:
        modularity_term = -torch.trace(c_probs.T @ modularity_matrix @ c_probs) / (2.0 * m)
    collapse = np.sqrt(k / max(n, 1)) * torch.norm(c_probs.sum(dim=0), p=2) - 1.0
    return modularity_term + collapse


def _predict_labels(z: torch.Tensor, c_probs: torch.Tensor, num_clusters: int, head: str) -> np.ndarray:
    z_np = z.detach().cpu().numpy()
    c_np = c_probs.detach().cpu().numpy()
    if head == "argmax":
        return np.argmax(c_np, axis=1).astype(np.int64)
    if head == "kmeans":
        km = KMeans(n_clusters=num_clusters, n_init=20, random_state=0).fit(z_np)
        return km.labels_.astype(np.int64)
    raise ValueError("Unsupported cluster head: {h}".format(h=head))


def train_gnn_dmon(
    x_init: torch.Tensor,
    edge_index: torch.Tensor,
    num_clusters: int,
    num_epochs: int = 4000,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    eval_interval: int = 20,
    loss_patience: int = 80,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    cluster_head: str = "kmeans",
    backbone_mode: str = "paper",
    device: str = "cuda",
) -> Dict:
    x = x_init.to(device)
    edge_index = edge_index.to(device)
    n = x.size(0)
    in_dim = x.size(1)

    if backbone_mode == "decoupled":
        model: nn.Module = ResidualGCNDecoupled(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_clusters=num_clusters,
            dropout=dropout,
        ).to(device)
    else:
        model = ResidualGCNPaper(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_clusters,
            dropout=dropout,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    b, m = _build_modularity_matrix(edge_index=edge_index, num_nodes=n, device=device)

    best_loss = None
    best_state = None
    stale = 0

    for epoch in range(num_epochs):
        model.train()
        z, c_probs = model(x, edge_index)
        loss = _dmon_loss(c_probs, b, m)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_interval == 0:
            loss_value = float(loss.detach().cpu().item())
            if best_loss is None or loss_value < best_loss - 1e-6:
                best_loss = loss_value
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= loss_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z, c_probs = model(x, edge_index)
    y_pred = _predict_labels(z, c_probs, num_clusters=num_clusters, head=cluster_head)

    return {
        "embeddings": z.detach().cpu().numpy().astype(np.float32),
        "pred_labels": y_pred,
        "best_loss": float(best_loss if best_loss is not None else 0.0),
    }
