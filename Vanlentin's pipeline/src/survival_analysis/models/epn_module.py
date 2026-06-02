#!/usr/bin/env python3
"""
EPNModule: EPN_surv wrapped as a PyTorch Lightning module.

The MLP is pre-trained and frozen; its survival-curve predictions are
already embedded in the DataFrames produced by EPNDataModule.
EPN only learns the attention-based correction on top of those predictions.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import torch
import lightning as pl

# Add surv_epn root to sys.path so EPN_surv's internal imports resolve.
_SURV_EPN_PATH = Path(__file__).parent.parent.parent.parent.parent / "surv_epn"
if str(_SURV_EPN_PATH) not in sys.path:
    sys.path.insert(0, str(_SURV_EPN_PATH))

from functions.epn import EPN_surv  # noqa: E402


class EPNModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper around EPN_surv.

    Expects each batch to be a pandas DataFrame produced by EPNDataModule.
    The survival-curve columns contain pre-computed MLP predictions.
    """

    def __init__(
        self,
        n_feats: int,
        timepoints: List[float],
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        alpha: float = 0.0,
        beta: float = 0.0,
        n_control: int = 1,
        prop_neighbors: float = 0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["timepoints"])
        self._timepoints = torch.tensor(timepoints, dtype=torch.float32)
        self._lr = learning_rate
        self._wd = weight_decay

        params = SimpleNamespace(
            feats_for_popg="METABRIC",
            feats_usable_surv=[],
            feats_usable_surv_dim=[],
        )
        self.epn = EPN_surv(
            _parameters=params,
            alpha=alpha,
            beta=beta,
            n_control=n_control,
            input_size=n_feats,
            prop_neighbors=prop_neighbors,
        )

    def training_step(self, batch: pd.DataFrame, batch_idx: int) -> torch.Tensor:
        corrected = self.epn(batch)  # [n_computed, T]

        computed_mask = batch.eval("pat_to_compute and batch").values
        labels = torch.tensor(
            batch[computed_mask][["label_duration", "label_event"]].values,
            dtype=torch.float32,
        )

        loss = self.epn.loss(self._timepoints, corrected, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch: pd.DataFrame, batch_idx: int) -> None:
        with torch.no_grad():
            corrected = self.epn(batch)
        self.log("val/n_preds", float(corrected.shape[0]),
                 on_step=False, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.epn.parameters(),
            lr=self._lr,
            weight_decay=self._wd,
        )
