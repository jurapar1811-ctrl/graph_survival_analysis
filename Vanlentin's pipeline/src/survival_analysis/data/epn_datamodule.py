#!/usr/bin/env python3
"""
EPNDataModule: prepares pandas DataFrames for EPN_surv by pre-computing
MLP survival curves and building the expected DataFrame format.

EPN_surv.forward() expects a DataFrame with:
  - numeric string columns (e.g. "0.0", "120.5") holding MLP survival curves
  - feat_0 .. feat_{n-1} holding patient features
  - management columns: label_duration, label_event, split, pat_to_compute, batch, pat_id
"""

from typing import Optional

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from pycox.models import CoxPH
from torch.utils.data import DataLoader, Dataset


class DataFrameDataset(Dataset):
    """Single-item dataset wrapping one DataFrame."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> pd.DataFrame:
        return self._df


class MiniBatchEPNDataset(Dataset):
    """Mini-batch dataset for EPN training.

    Each item is the full training DataFrame, but only `batch_size`
    randomly-selected patients have pat_to_compute=True / batch=True.
    All other patients stay in the DataFrame as the neighbour pool.

    This caps the intermediate [n_batch, n_batch, T] tensor to a safe size
    instead of allocating [n_train, n_train, T] all at once.
    """

    def __init__(self, train_df: pd.DataFrame, batch_size: int = 256) -> None:
        self._df = train_df.copy()
        self._batch_size = batch_size
        self._all_idx = train_df.index.tolist()
        # expose roughly one pass over all patients per epoch
        self._n_batches = max(1, len(self._all_idx) // batch_size)

    def __len__(self) -> int:
        return self._n_batches

    def __getitem__(self, idx: int) -> pd.DataFrame:
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
        return df


def df_collate_fn(batch):
    return batch[0]


class EPNDataModule(LightningDataModule):
    """
    Wraps MetabricGraphSurvivalDataModule + trained MLPModule.
    Pre-computes MLP survival curves in setup() so EPN training
    does not need to call CoxPH at every step.
    """

    def __init__(
        self,
        base_datamodule: LightningDataModule,
        mlp_module,
    ) -> None:
        super().__init__()
        self._base = base_datamodule
        self._mlp = mlp_module

    def setup(self, stage: Optional[str] = None) -> None:
        self._base.setup(stage)

        train_x = self._base.train_graph.x
        train_dur = self._base.train_graph.y[..., 0]
        train_evt = self._base.train_graph.y[..., 1]
        all_x = self._base.val_graph.x       # all patients
        all_y = self._base.val_graph.y       # all patients
        val_idx = self._base.val_graph.val_idx.numpy()

        # --- Compute MLP survival curves for all patients ---
        self._mlp.eval()
        with torch.no_grad():
            cox = CoxPH(self._mlp)
            cox.compute_baseline_hazards(train_x, (train_dur, train_evt))
            surv_df = cox.predict_surv_df(all_x)  # [T x N_all]

        n_all = all_x.shape[0]
        n_feats = all_x.shape[1]
        all_y_np = all_y.numpy()
        surv_T = surv_df.T  # [N_all x T], columns = float time points

        self.timepoints = sorted(surv_T.columns.astype(float).tolist())
        self.n_feats = n_feats

        # --- Build feature columns ---
        feat_data = {f"feat_{i}": all_x[:, i].numpy() for i in range(n_feats)}

        # --- Build survival-curve columns (string keys expected by EPN) ---
        tp_data = {str(float(t)): surv_T[t].values for t in self.timepoints}

        # --- Split labels ---
        split_col = np.array(["train"] * n_all, dtype=object)
        split_col[val_idx] = "val"

        # --- Assemble base DataFrame ---
        base_dict: dict = {
            "pat_id": np.arange(n_all),
            "label_duration": all_y_np[:, 0],
            "label_event": all_y_np[:, 1],
            "split": split_col,
        }
        base_dict.update(feat_data)
        base_dict.update(tp_data)
        base_df = pd.DataFrame(base_dict).reset_index(drop=True)

        # --- Train DataFrame: training patients only (batch flags set per mini-batch) ---
        train_mask = base_df["split"] == "train"
        train_df = base_df[train_mask].copy().reset_index(drop=True)
        # pat_to_compute / batch are assigned dynamically in MiniBatchEPNDataset
        train_df["pat_to_compute"] = False
        train_df["batch"] = False

        # --- Val DataFrame: all patients; val patients are the compute targets ---
        val_df = base_df.copy()
        val_df["pat_to_compute"] = False
        val_df["batch"] = False
        val_mask = val_df["split"] == "val"
        val_df.loc[val_mask, "pat_to_compute"] = True
        val_df.loc[val_mask, "batch"] = True

        self.train_df = train_df
        self.val_df = val_df

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            MiniBatchEPNDataset(self.train_df, batch_size=256),
            batch_size=1,
            collate_fn=df_collate_fn,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            DataFrameDataset(self.val_df),
            batch_size=1,
            collate_fn=df_collate_fn,
            shuffle=False,
        )
