#!/usr/bin/env python3
"""
Her MLP model wrapped as a PyTorch Lightning module.

Architecture mirrors surv_epn/functions/models.py::MLP (two hidden layers,
ReLU activations) and is trained with the CoxPH partial-likelihood loss so
that it is directly comparable to SurvivalDGM.
"""

from typing import Callable

import torch
import torch.nn as nn
import lightning as pl
from pycox.models.loss import CoxPHLoss


class MLPModule(pl.LightningModule):
    """
    Two-hidden-layer MLP for survival analysis, trained with CoxPH loss.

    Designed to consume the same DataLoader format produced by
    MetabricGraphSurvivalDataModule (PyG Data objects where
    batch.x = features, batch.y = [duration, event]).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        optimizer: Callable,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer"])
        self._optimizer = optimizer

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        self.loss_fn = CoxPHLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        pred = self(batch.x)
        durations, events = batch.y[..., 0], batch.y[..., 1]
        loss = self.loss_fn(pred, durations, events)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        all_pred = self(batch.x)
        pred = all_pred[batch.val_idx]
        durations = batch.y[batch.val_idx, 0]
        events = batch.y[batch.val_idx, 1]
        loss = self.loss_fn(pred, durations, events)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return self._optimizer(self.parameters())
