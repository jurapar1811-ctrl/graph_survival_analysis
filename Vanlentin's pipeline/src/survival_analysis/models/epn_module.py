#!/usr/bin/env python3
"""
EPNModule — graph-based Error Propagation Network as a Lightning module.

Design principle (per supervisor feedback):
  Models should depend only on hyperparameters (dimensions, k, etc.),
  NOT on DataFrames or dataset-specific objects.

EPNModel.forward() takes pure tensors:
    features         [N, F]      patient feature vectors
    surv_preds       [N, T]      MLP baseline survival curves
    labels           [N, 2]      (duration, event)
    nn_idx           [N, k]      precomputed k-NN neighbour matrix
    test_idx         [n_batch]   indices of patients to compute for
    val_patient_idx  [n_val]|None  val patient indices (for attention masking)

No DataFrames, no column-name parsing, no surv_epn imports.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchtuples as tt
from pycox.models.loss import CoxCCLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _survival_to_hazard(pred: torch.Tensor) -> torch.Tensor:
    """Convert survival curve [N, T] to discrete hazard [N, T]."""
    cum_hazard = -torch.log(torch.clamp(pred, min=1e-9))
    return torch.diff(cum_hazard, prepend=torch.zeros(pred.shape[0], 1))


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class EPNModel(nn.Module):
    """
    Graph-constrained Error Propagation Network.

    Hyperparameters only in __init__ — no data objects.
    """

    def __init__(
        self,
        input_size: int,
        timepoints: List[float],
        temp_att: float = 1.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        n_control: int = 1,
    ) -> None:
        super().__init__()
        self._temperature  = temp_att
        self._alpha        = alpha
        self._beta         = beta
        self._n_control    = n_control

        self.register_buffer(
            "_timepoints", torch.tensor(timepoints, dtype=torch.float32)
        )
        self._dk = torch.sqrt(torch.tensor(float(input_size)))

        self._key_projection   = nn.Linear(input_size, input_size)
        self._query_projection = nn.Linear(input_size, input_size)
        self._criterion        = CoxCCLoss(shrink=0.0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        features: torch.Tensor,
        surv_preds: torch.Tensor,
        labels: torch.Tensor,
        nn_idx: torch.Tensor,
        test_idx: torch.Tensor,
        val_patient_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return graph-corrected survival curves for test_idx patients [n_batch, T]."""
        F_dim   = features.shape[1]
        n_batch = len(test_idx)
        k       = nn_idx.shape[1]

        errors       = self._compute_error_survival(surv_preds, labels)  # [N, T]
        neighbour_idx = nn_idx[test_idx]                                 # [n_batch, k]

        q    = self._query_projection(features[test_idx])
        keys = self._key_projection(
            features[neighbour_idx].view(-1, F_dim)
        ).view(n_batch, k, F_dim)

        att = torch.bmm(q.unsqueeze(1), keys.permute(0, 2, 1)).squeeze(1) / self._dk

        if val_patient_idx is not None:
            att = att.masked_fill(
                torch.isin(neighbour_idx, val_patient_idx),
                torch.finfo(torch.float32).min,
            )
        else:
            att = att.masked_fill(
                neighbour_idx == test_idx.unsqueeze(1),
                torch.finfo(torch.float32).min,
            )

        att        = F.softmax(att * self._temperature, dim=-1)
        correction = torch.einsum("bk,bkt->bt", att, errors[neighbour_idx])
        return surv_preds[test_idx] + correction

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """CoxCC loss + elastic-net penalty."""
        l1, l2 = torch.tensor(0.0), torch.tensor(0.0)
        for _, w in self.named_parameters():
            l1 = l1 + w.abs().sum()
            l2 = l2 + w.pow(2).sum()

        g_case, g_control = self._make_case_control(
            self._timepoints, _survival_to_hazard(pred), y
        )
        return (
            self._criterion(g_case, g_control)
            + self._alpha * l1
            + self._beta  * l2
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_error_survival(
        self, pred: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        tp = self._timepoints
        T, N = tp.shape[0], y.shape[0]
        cont = torch.where(
            tp.repeat(N, 1) > y[:, 0].repeat(T, 1).t(),
            torch.zeros(N, T), torch.ones(N, T),
        )
        nan_fill = torch.full((N, T), float("nan"))
        cont = torch.where(
            (y[:, 1] == 1.0).repeat(T, 1).t(),
            cont,
            torch.where(cont.bool(), torch.ones(N, T), nan_fill),
        )
        return torch.where(cont.isnan(), torch.zeros_like(cont), cont - pred)

    @staticmethod
    def _sorted_input_target(pred, y):
        idx = np.argsort(y[:, 0].detach().cpu().numpy())
        if (idx == np.arange(len(idx))).all():
            return pred, y
        t = torch.tensor(idx)
        return pred[t], y[t]

    @staticmethod
    def _make_at_risk_dict(durations):
        import pandas as _pd
        s = _pd.Series(durations)
        keys = s.drop_duplicates(keep="first")
        return {t: s.index.values[ix:] for ix, t in keys.items()}

    @staticmethod
    def _sample_alive(dates, at_risk, n_control):
        lengths = np.array([at_risk[x].shape[0] for x in dates])
        idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype(int)
        samp = np.empty((dates.size, n_control), dtype=int)
        for it, t in enumerate(dates):
            samp[it, :] = at_risk[t][idx[:, it]]
        return samp

    def _make_case_control(self, timepoints, pred_hazard, y):
        s_pred, s_y = self._sorted_input_target(pred_hazard, y)
        at_risk  = self._make_at_risk_dict(s_y[:, 0].detach().cpu().numpy())
        tp_np    = timepoints.detach().cpu().numpy()

        idx_event = torch.where(s_y[:, 1] == 1.0)[0]
        tp_case   = np.where([tp_np == s_y[i, 0].item() for i in idx_event])[1]
        idx_case  = [[idx_event[i].item(), tp_case[i]] for i in range(len(idx_event))]
        g_case    = torch.stack([s_pred[r, c] for r, c in idx_case])

        ctrl_idx  = self._sample_alive(
            s_y[idx_event, 0].detach().cpu().numpy(), at_risk, self._n_control
        )
        ctrl_fmt  = [
            [[ctrl_idx[i, j], tp_case[i]] for i in range(len(idx_event))]
            for j in range(self._n_control)
        ]
        controls  = [
            torch.stack([s_pred[r, c] for r, c in ctrl_fmt[j]])
            for j in range(self._n_control)
        ]
        g_control = (
            tt.TupleTree(controls[0].unsqueeze(0))
            if self._n_control == 1
            else tt.TupleTree(tuple(controls))
        )
        return g_case, g_control


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------

class EPNModule(pl.LightningModule):
    """Lightning wrapper for EPNModel. Batches are tensor dicts from EPNDataModule."""

    def __init__(
        self,
        n_feats: int,
        timepoints: List[float],
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        alpha: float = 0.0,
        beta: float = 0.0,
        n_control: int = 1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["timepoints"])
        self._lr = learning_rate
        self._wd = weight_decay

        self.epn = EPNModel(
            input_size=n_feats,
            timepoints=timepoints,
            alpha=alpha,
            beta=beta,
            n_control=n_control,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        corrected = self.epn(
            batch["features"], batch["surv_preds"], batch["labels"],
            batch["nn_idx"],   batch["test_idx"],   batch["val_patient_idx"],
        )
        loss = self.epn.loss(corrected, batch["labels"][batch["test_idx"]])
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        with torch.no_grad():
            corrected = self.epn(
                batch["features"], batch["surv_preds"], batch["labels"],
                batch["nn_idx"],   batch["test_idx"],   batch["val_patient_idx"],
            )
        self.log("val/n_preds", float(corrected.shape[0]),
                 on_step=False, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.epn.parameters(), lr=self._lr, weight_decay=self._wd
        )
