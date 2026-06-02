#!/usr/bin/env python3
"""
Standalone patient similarity graph construction function.

The supervisor's instruction was to first build this graph function independently
(using PyTorch Geometric), test it, and then move on to error-propagation training
(EPN). This matches the original EPN paper's concept: patients that are similar in
feature space should share survival error corrections.

How it works
------------
1. Compute pairwise similarity between patients based on their clinical features.
2. For each patient, keep the k most similar patients as neighbours.
3. Return a torch_geometric.data.Data object where nodes = patients, edges = similarity links.

Comparison with Oriane's (EPN_surv) approach
---------------------------------------------
Oriane's EPN_surv builds the similarity graph *implicitly* inside the model using a
learned attention mechanism.  This function builds an *explicit* k-NN graph before
training, which can be inspected and visualised independently.
"""

from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree


def build_patient_similarity_graph(
    features: torch.Tensor,
    k: int = 10,
    metric: Literal["cosine", "euclidean"] = "cosine",
    loop: bool = False,
) -> Data:
    """
    Build a patient similarity graph using k-NN.

    Args:
        features:  [N, D] float tensor of patient features (already normalised).
        k:         Number of nearest neighbours per patient.
        metric:    Similarity metric — "cosine" or "euclidean".
        loop:      If True, include self-loops (patient -> itself).

    Returns:
        torch_geometric.data.Data with:
            x          : [N, D] patient features
            edge_index : [2, E] directed edges — each patient points to its k neighbours
            num_nodes  : N
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D [N, D], got shape {features.shape}")

    n_patients = features.shape[0]
    if k >= n_patients:
        raise ValueError(
            f"k={k} must be less than the number of patients ({n_patients})"
        )

    if metric == "cosine":
        normed = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        # similarity[i,j] = cosine similarity between patient i and patient j
        sim = normed @ normed.T           # [N, N]
        if not loop:
            sim.fill_diagonal_(float("-inf"))
        # Higher similarity = closer neighbour
        _, nn_idx = sim.topk(k, dim=1)   # [N, k]  nn_idx[i] = k neighbours of i

    elif metric == "euclidean":
        # dist[i,j] = squared Euclidean distance between i and j
        diff = features.unsqueeze(0) - features.unsqueeze(1)   # [N, N, D]
        dist = (diff ** 2).sum(dim=-1)                          # [N, N]
        if not loop:
            dist.fill_diagonal_(float("inf"))
        # Smaller distance = closer neighbour
        _, nn_idx = dist.topk(k, dim=1, largest=False)          # [N, k]

    else:
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got '{metric}'")

    # Build edge_index: edge (neighbour -> patient) i.e. source=neighbour, target=patient
    # nn_idx[i, j] = j-th nearest neighbour of patient i  →  edge from nn_idx[i,j] to i
    targets = torch.arange(n_patients).unsqueeze(1).expand(-1, k).reshape(-1)  # [N*k]
    sources = nn_idx.reshape(-1)                                                 # [N*k]
    edge_index = torch.stack([sources, targets], dim=0)                          # [2, N*k]

    return Data(x=features, edge_index=edge_index, num_nodes=n_patients)


def describe_graph(graph: Data) -> None:
    """Print a short summary of the graph for manual verification."""
    n = graph.num_nodes
    e = graph.edge_index.shape[1]
    deg = degree(graph.edge_index[1], num_nodes=n)
    print(f"Nodes         : {n}")
    print(f"Edges         : {e}")
    print(f"Avg degree    : {e / n:.1f}")
    print(f"Feature dim   : {graph.x.shape[1]}")
    print(f"Degree min/max: {int(deg.min())} / {int(deg.max())}")
