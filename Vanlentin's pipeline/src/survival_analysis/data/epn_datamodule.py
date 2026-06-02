#!/usr/bin/env python3
"""
EPNDataModule: prepares DataFrames + k-NN neighbor matrices for graph-based EPN.

Each batch yielded by the DataLoaders is a dict:
    {
        "df"     : pd.DataFrame  (patient features, MLP survival curves, mgmt cols)
        "nn_idx" : torch.Tensor  [N, k]  precomputed k-NN neighbour indices
    }

The k-NN graph is built once in setup() using build_patient_similarity_graph().
EPNModel uses nn_idx to restrict attention to the k nearest neighbours instead of
recomputing similarity across all patients at every step.
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from pycox.models import CoxPH
from torch.utils.data import DataLoader, Dataset

from survival_analysis.data.graph_builder import build_patient_similarity_graph


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class GraphEPNDataset(Dataset):
    """Single-item dataset that returns a (df, nn_idx) pair."""

    def __init__(self, df: pd.DataFrame, nn_idx: torch.Tensor) -> None:
        self._df = df
        self._nn_idx = nn_idx

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return {"df": self._df, "nn_idx": self._nn_idx}


class MiniBatchEPNDataset(Dataset):
    """Mini-batch dataset for EPN training with graph-based neighbours.

    Each item is the full training DataFrame with a random subset of
    `batch_size` patients marked as the current batch, together with the
    precomputed training-graph neighbour matrix.

    Using the graph (k neighbours) instead of full attention caps the
    intermediate tensor at [batch_size, k, T] instead of [N_train, N_train, T].
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        train_nn_idx: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        self._df = train_df.copy()
        self._nn_idx = train_nn_idx
        self._batch_size = batch_size
        self._all_idx = train_df.index.tolist()
        self._n_batches = max(1, len(self._all_idx) // batch_size)

    def __len__(self) -> int:
        return self._n_batches

    def __getitem__(self, idx: int) -> dict:
        df = self._df.copy()
        df["pat_to_compute"] = False
        df["batch"] = False
        selected = np.random.choice(
            self._all_idx,
            size=min(self._batch_size, len(self._all_idx)),
            replace=False,
        )
        df.loc[selected, "pat_to_compute"] = True
        df.loc[selected, "batch"] = True
        return {"df": df, "nn_idx": self._nn_idx}


def graph_collate_fn(batch: list) -> dict:
    """Return the single-item batch dict unchanged."""
    return batch[0]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class EPNDataModule(LightningDataModule):
    """
    Wraps MetabricGraphSurvivalDataModule + trained MLPModule.

    In setup():
      1. Pre-compute MLP survival curves for all patients (CoxPH wrapper).
      2. Build k-NN patient-similarity graphs for training and validation sets.
      3. Store train_df / val_df DataFrames and train_nn_idx / val_nn_idx tensors.
    """

    def __init__(
        self,
        base_datamodule: LightningDataModule,
        mlp_module,
        graph_k: int = 10,
        graph_metric: str = "cosine",
    ) -> None:
        super().__init__()
        self._base = base_datamodule
        self._mlp = mlp_module
        self._graph_k = graph_k
        self._graph_metric = graph_metric

    def setup(self, stage: Optional[str] = None) -> None:
        self._base.setup(stage)

        train_x = self._base.train_graph.x
        train_dur = self._base.train_graph.y[..., 0]
        train_evt = self._base.train_graph.y[..., 1]
        all_x = self._base.val_graph.x
        all_y = self._base.val_graph.y
        val_idx = self._base.val_graph.val_idx.numpy()

        # --- 1. Compute MLP survival curves ---
        self._mlp.eval()
        with torch.no_grad():
            cox = CoxPH(self._mlp)
            cox.compute_baseline_hazards(train_x, (train_dur, train_evt))
            surv_df = cox.predict_surv_df(all_x)   # [T × N_all]

        n_all = all_x.shape[0]
        n_feats = all_x.shape[1]
        all_y_np = all_y.numpy()
        surv_T = surv_df.T

        self.timepoints = sorted(surv_T.columns.astype(float).tolist())
        self.n_feats = n_feats

        # --- 2. Assemble base DataFrame ---
        feat_data = {f"feat_{i}": all_x[:, i].numpy() for i in range(n_feats)}
        tp_data = {str(float(t)): surv_T[t].values for t in self.timepoints}
        split_col = np.array(["train"] * n_all, dtype=object)
        split_col[val_idx] = "val"

        base_dict: dict = {
            "pat_id": np.arange(n_all),
            "label_duration": all_y_np[:, 0],
            "label_event": all_y_np[:, 1],
            "split": split_col,
        }
        base_dict.update(feat_data)
        base_dict.update(tp_data)
        base_df = pd.DataFrame(base_dict).reset_index(drop=True)

        # --- 3. Train / Val DataFrames ---
        train_mask = base_df["split"] == "train"
        train_df = base_df[train_mask].copy().reset_index(drop=True)
        train_df["pat_to_compute"] = False
        train_df["batch"] = False

        val_df = base_df.copy()
        val_df["pat_to_compute"] = False
        val_df["batch"] = False
        val_df.loc[val_df["split"] == "val", "pat_to_compute"] = True
        val_df.loc[val_df["split"] == "val", "batch"] = True

        self.train_df = train_df
        self.val_df = val_df

        # --- 4. Build k-NN graphs ---
        # Training graph: only training patient features
        feat_cols = [f"feat_{i}" for i in range(n_feats)]
        train_feats = torch.tensor(train_df[feat_cols].values, dtype=torch.float32)
        train_graph = build_patient_similarity_graph(
            train_feats, k=self._graph_k, metric=self._graph_metric, loop=False
        )
        self.train_nn_idx = self._edge_index_to_nn_matrix(
            train_graph.edge_index, n_nodes=len(train_df), k=self._graph_k
        )

        # Validation graph: all patients (val patients attend to train patients)
        all_feats = torch.tensor(
            val_df[feat_cols].values, dtype=torch.float32
        )
        val_graph = build_patient_similarity_graph(
            all_feats, k=self._graph_k, metric=self._graph_metric, loop=False
        )
        self.val_nn_idx = self._edge_index_to_nn_matrix(
            val_graph.edge_index, n_nodes=len(val_df), k=self._graph_k
        )

        print(
            f"  Graphs built — train: {len(train_df)} nodes × {self._graph_k} neighbours, "
            f"val: {len(val_df)} nodes × {self._graph_k} neighbours"
        )

    # ------------------------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MiniBatchEPNDataset(self.train_df, self.train_nn_idx, batch_size=256),
            batch_size=1,
            collate_fn=graph_collate_fn,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            GraphEPNDataset(self.val_df, self.val_nn_idx),
            batch_size=1,
            collate_fn=graph_collate_fn,
            shuffle=False,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _edge_index_to_nn_matrix(
        edge_index: torch.Tensor, n_nodes: int, k: int
    ) -> torch.Tensor:
        """Convert edge_index [2, N*k] to neighbour matrix [N, k].

        edge_index[0] = source (neighbour), edge_index[1] = target (patient).
        """
        src, tgt = edge_index[0], edge_index[1]
        nn_matrix = torch.zeros(n_nodes, k, dtype=torch.long)
        for i in range(n_nodes):
            mask = tgt == i
            nn_matrix[i] = src[mask][:k]
        return nn_matrix
