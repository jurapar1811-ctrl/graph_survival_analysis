#!/usr/bin/env python3
"""
EPNDataModule — tensor-only interface.

Each DataLoader batch is a plain dict of tensors:
    {
        "features"       : Tensor [N, F]      patient features
        "surv_preds"     : Tensor [N, T]      MLP survival curves
        "labels"         : Tensor [N, 2]      (duration, event)
        "nn_idx"         : Tensor [N, k]      k-NN neighbour indices
        "test_idx"       : Tensor [n_batch]   indices of the current batch patients
        "val_patient_idx": Tensor | None      val patient indices (None during training)
    }

Models receive only tensors — no DataFrames, no column-name parsing.
"""

from typing import Optional

import numpy as np
import torch
from lightning import LightningDataModule
from pycox.models import CoxPH
from torch.utils.data import DataLoader, Dataset

from survival_analysis.data.graph_builder import build_patient_similarity_graph


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TrainEPNDataset(Dataset):
    """Yields one tensor-dict per step with a random subset of training patients."""

    def __init__(
        self,
        features: torch.Tensor,
        surv_preds: torch.Tensor,
        labels: torch.Tensor,
        nn_idx: torch.Tensor,
        batch_size: int = 256,
    ) -> None:
        self._features   = features
        self._surv_preds = surv_preds
        self._labels     = labels
        self._nn_idx     = nn_idx
        self._n          = len(features)
        self._batch_size = batch_size
        self._n_batches  = max(1, self._n // batch_size)

    def __len__(self) -> int:
        return self._n_batches

    def __getitem__(self, idx: int) -> dict:
        test_idx = torch.randperm(self._n)[: self._batch_size]
        return {
            "features":        self._features,
            "surv_preds":      self._surv_preds,
            "labels":          self._labels,
            "nn_idx":          self._nn_idx,
            "test_idx":        test_idx,
            "val_patient_idx": None,
        }


class ValEPNDataset(Dataset):
    """Single-item dataset holding the full val-set tensors."""

    def __init__(
        self,
        all_features: torch.Tensor,
        all_surv_preds: torch.Tensor,
        all_labels: torch.Tensor,
        val_nn_idx: torch.Tensor,
        val_patient_idx: torch.Tensor,
    ) -> None:
        self._batch = {
            "features":        all_features,
            "surv_preds":      all_surv_preds,
            "labels":          all_labels,
            "nn_idx":          val_nn_idx,
            "test_idx":        val_patient_idx,
            "val_patient_idx": val_patient_idx,
        }

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        return self._batch


def tensor_collate_fn(batch: list) -> dict:
    return batch[0]


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------

class EPNDataModule(LightningDataModule):
    """
    Wraps MetabricGraphSurvivalDataModule + trained MLPModule.

    After setup():
      - train_features / train_surv_preds / train_labels / train_nn_idx
      - all_features  / all_surv_preds   / all_labels   / val_nn_idx
      - val_patient_idx : LongTensor of val patient positions in all_* arrays
      - timepoints      : list[float] — survival curve time points
    """

    def __init__(
        self,
        base_datamodule: LightningDataModule,
        mlp_module,
        graph_k: int = 10,
        graph_metric: str = "cosine",
    ) -> None:
        super().__init__()
        self._base         = base_datamodule
        self._mlp          = mlp_module
        self._graph_k      = graph_k
        self._graph_metric = graph_metric

    def setup(self, stage: Optional[str] = None) -> None:
        self._base.setup(stage)

        train_x   = self._base.train_graph.x
        train_dur = self._base.train_graph.y[..., 0]
        train_evt = self._base.train_graph.y[..., 1]
        all_x     = self._base.val_graph.x
        all_y     = self._base.val_graph.y
        val_idx   = self._base.val_graph.val_idx          # LongTensor [n_val]

        # ── 1. MLP survival curves ──────────────────────────────────────────
        self._mlp.eval()
        with torch.no_grad():
            cox = CoxPH(self._mlp)
            cox.compute_baseline_hazards(train_x, (train_dur, train_evt))
            surv_df = cox.predict_surv_df(all_x)          # [T × N_all]

        self.timepoints = sorted(surv_df.index.astype(float).tolist())

        surv_np  = surv_df.values.T.astype(np.float32)   # [N_all, T]
        all_surv = torch.from_numpy(surv_np)

        # determine training patient indices inside all_x
        n_all      = all_x.shape[0]
        train_mask = torch.ones(n_all, dtype=torch.bool)
        train_mask[val_idx] = False
        train_indices = torch.where(train_mask)[0]        # [N_train]

        # ── 2. Store tensors ────────────────────────────────────────────────
        self.all_features    = all_x                      # [N_all, F]
        self.all_surv_preds  = all_surv                   # [N_all, T]
        self.all_labels      = all_y                      # [N_all, 2]
        self.val_patient_idx = val_idx                    # [n_val]

        self.train_features   = all_x[train_indices]      # [N_train, F]
        self.train_surv_preds = all_surv[train_indices]   # [N_train, T]
        self.train_labels     = all_y[train_indices]      # [N_train, 2]

        self.n_feats = all_x.shape[1]

        # ── 3. Build k-NN graphs ────────────────────────────────────────────
        train_graph = build_patient_similarity_graph(
            self.train_features, k=self._graph_k,
            metric=self._graph_metric, loop=False,
        )
        self.train_nn_idx = self._edge_index_to_nn_matrix(
            train_graph.edge_index, len(train_indices), self._graph_k
        )

        val_graph = build_patient_similarity_graph(
            self.all_features, k=self._graph_k,
            metric=self._graph_metric, loop=False,
        )
        self.val_nn_idx = self._edge_index_to_nn_matrix(
            val_graph.edge_index, n_all, self._graph_k
        )

        print(
            f"  EPNDataModule ready — "
            f"train: {len(train_indices)} pts, val: {len(val_idx)} pts, "
            f"T={len(self.timepoints)}, k={self._graph_k}"
        )

    # ── DataLoaders ─────────────────────────────────────────────────────────

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            TrainEPNDataset(
                self.train_features, self.train_surv_preds,
                self.train_labels, self.train_nn_idx, batch_size=256,
            ),
            batch_size=1, collate_fn=tensor_collate_fn, shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            ValEPNDataset(
                self.all_features, self.all_surv_preds, self.all_labels,
                self.val_nn_idx, self.val_patient_idx,
            ),
            batch_size=1, collate_fn=tensor_collate_fn, shuffle=False,
        )

    # ── Helper ───────────────────────────────────────────────────────────────

    @staticmethod
    def _edge_index_to_nn_matrix(
        edge_index: torch.Tensor, n_nodes: int, k: int
    ) -> torch.Tensor:
        src, tgt = edge_index[0], edge_index[1]
        nn_matrix = torch.zeros(n_nodes, k, dtype=torch.long)
        for i in range(n_nodes):
            nn_matrix[i] = src[tgt == i][:k]
        return nn_matrix
