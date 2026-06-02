#!/usr/bin/env python3
"""
EPNModule: Error Propagation Network wrapped as a PyTorch Lightning module.

All EPN logic is implemented here from scratch — no imports from surv_epn.
surv_epn/ is kept in the workspace as a reference only.

Core idea: given MLP survival-curve predictions for all patients, EPN
learns an attention mechanism that corrects each patient's prediction by
looking at similar patients' prediction errors.
"""

from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import torchtuples as tt
from pycox.models.loss import CoxCCLoss


# ---------------------------------------------------------------------------
# Standalone helpers (mirror of surv_epn/functions/survival.py and epn.py)
# ---------------------------------------------------------------------------

def _get_hazard_from_survival_curve(pred: torch.Tensor) -> torch.Tensor:
    """Convert survival curve [N, T] to discrete hazard [N, T]."""
    cum_hazard = -torch.log(torch.clamp(pred, min=1e-9))
    return torch.diff(cum_hazard, prepend=torch.zeros(cum_hazard.shape[0], 1))


def _compute_attention(
    key_proj: nn.Linear,
    query_proj: nn.Linear,
    dk: torch.Tensor,
    x_query: torch.Tensor,
    x_key: torch.Tensor,
) -> torch.Tensor:
    """Scaled dot-product attention (learned projections)."""
    return torch.matmul(key_proj(x_query), query_proj(x_key).t()) / dk


# ---------------------------------------------------------------------------
# Core EPN model (no surv_epn dependency)
# ---------------------------------------------------------------------------

class EPNModel(nn.Module):
    """
    Error Propagation Network for survival analysis.

    Input: pandas DataFrame with
      - string time-point columns (MLP survival curves)
      - feat_X feature columns
      - management columns (split, pat_to_compute, batch, label_*)

    Output: corrected survival curves [n_batch_patients, T]
    """

    _MGMT_COLS = {"label_duration", "label_event", "split",
                  "pat_to_compute", "batch", "pat_id", "index"}

    def __init__(
        self,
        input_size: int,
        prop_neighbors: float = 0.1,
        temp_att: float = 1.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        n_control: int = 1,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._prop_neighbors = prop_neighbors
        self._temperature = temp_att
        self._alpha = alpha
        self._beta = beta
        self._n_control = n_control
        self._dk = torch.sqrt(torch.tensor(float(input_size)))

        self._key_projection = nn.Linear(input_size, input_size)
        self._query_projection = nn.Linear(input_size, input_size)
        self._criterion = CoxCCLoss(shrink=0.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, df: pd.DataFrame) -> torch.Tensor:
        """Return corrected survival curves for batch patients [n_batch, T]."""
        # Extract time-point columns (float-parseable, not feat_* or mgmt)
        tp_cols_sorted = self._get_timepoint_cols(df)
        preds = torch.from_numpy(df[tp_cols_sorted].values).float()          # [N, T]
        labels = torch.from_numpy(
            df[["label_duration", "label_event"]].values
        ).float()                                                              # [N, 2]
        feat_cols = [c for c in df.columns if "feat_" in c]
        features = torch.from_numpy(df[feat_cols].values).float()            # [N, F]

        test_idx = np.where(df.eval("pat_to_compute and batch"))[0]
        training = (
            len(np.where(df.eval(
                'pat_to_compute and (split=="val" or split=="test")'
            ))[0]) == 0
        )

        errors = self._compute_error_survival(
            torch.tensor([float(c) for c in tp_cols_sorted]), preds, labels
        )                                                                      # [N, T]

        att = _compute_attention(
            self._key_projection, self._query_projection, self._dk,
            features[test_idx], features,
        )                                                                      # [n_batch, N]

        if not training:
            full_test_idx = np.where(df["split"].values == "val")[0]
            att[:, full_test_idx] = torch.finfo(torch.float32).min
        else:
            att[range(len(test_idx)), test_idx] = torch.finfo(torch.float32).min

        if self._prop_neighbors != -1:
            k = max(round(self._prop_neighbors * att.shape[1]), 1)
            att, indices = torch.topk(att, k=k, dim=1)   # [n_batch, k]
            errors = errors[indices]                       # [n_batch, k, T]
        else:
            errors = (errors.unsqueeze(1)
                      .repeat(1, len(test_idx), 1)
                      .permute(1, 0, 2))                   # [n_batch, N, T]

        att = F.softmax(att * self._temperature, dim=-1)
        preds_test = preds[test_idx]                       # [n_batch, T]
        return torch.diagonal(torch.matmul(att, errors), 0).t() + preds_test

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        timepoints: torch.Tensor,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """CoxCC loss with elastic-net penalty."""
        l1, l2 = torch.tensor(0.0), torch.tensor(0.0)
        for _, w in self.named_parameters():
            l1 = l1 + w.abs().sum()
            l2 = l2 + w.pow(2).sum()

        pred_hazard = _get_hazard_from_survival_curve(pred)
        g_case, g_control = self._make_case_control(timepoints, pred_hazard, y)
        main_loss = self._criterion(g_case, g_control)
        return main_loss + self._alpha * l1 + self._beta * l2

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_error_survival(
        timepoints: torch.Tensor,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Error between predicted survival curve and step-function label [N, T]."""
        T = timepoints.shape[0]
        N = y.shape[0]
        # Ideal survival: 1.0 before event, 0.0 after (float from the start)
        cont = torch.where(
            timepoints.repeat(N, 1) > y[:, 0].repeat(T, 1).t(),
            torch.zeros(N, T), torch.ones(N, T),
        )
        # NaN at censored time-points (event=0)
        nan_fill = torch.full((N, T), float("nan"))
        cont = torch.where(
            (y[:, 1] == 1.0).repeat(T, 1).t(),
            cont,
            torch.where(cont.bool(), torch.ones(N, T), nan_fill),
        )
        return torch.where(cont.isnan(), torch.zeros_like(cont), cont - pred)

    @staticmethod
    def _sorted_input_target(
        pred: torch.Tensor, y: torch.Tensor
    ):
        durations_np = y[:, 0].detach().cpu().numpy()
        idx_sort = np.argsort(durations_np)
        if (idx_sort == np.arange(len(idx_sort))).all():
            return pred, y
        idx_t = torch.tensor(idx_sort)
        return pred[idx_t], y[idx_t]

    @staticmethod
    def _make_at_risk_dict(durations: np.ndarray) -> dict:
        s = pd.Series(durations)
        allidx = s.index.values
        keys = s.drop_duplicates(keep="first")
        return {t: allidx[ix:] for ix, t in keys.items()}

    @staticmethod
    def _sample_alive_from_dates(
        dates: np.ndarray, at_risk_dict: dict, n_control: int = 1
    ) -> np.ndarray:
        lengths = np.array([at_risk_dict[x].shape[0] for x in dates])
        idx = (np.random.uniform(size=(n_control, dates.size)) * lengths).astype(int)
        samp = np.empty((dates.size, n_control), dtype=int)
        for it, t in enumerate(dates):
            samp[it, :] = at_risk_dict[t][idx[:, it]]
        return samp

    def _make_case_control(
        self,
        timepoints: torch.Tensor,
        pred_hazard: torch.Tensor,
        y: torch.Tensor,
    ):
        """Build (g_case, g_control) tensors for CoxCCLoss."""
        s_pred, s_y = self._sorted_input_target(pred_hazard, y)
        at_risk = self._make_at_risk_dict(s_y[:, 0].detach().cpu().numpy())
        tp_np = timepoints.detach().cpu().numpy()

        idx_event = torch.where(s_y[:, 1] == 1.0)[0]
        tp_case = np.where(
            [tp_np == s_y[i, 0].item() for i in idx_event]
        )[1]

        # g_case: hazard at event time for each event patient
        idx_case = [[idx_event[i].item(), tp_case[i]] for i in range(len(idx_event))]
        g_case = torch.stack([s_pred[r, c] for r, c in idx_case])

        # g_control: n_control randomly sampled alive patients
        ctrl_idx = self._sample_alive_from_dates(
            s_y[idx_event, 0].detach().cpu().numpy(), at_risk, self._n_control
        )                                                   # [n_events, n_control]
        ctrl_fmt = [
            [[ctrl_idx[i, j], tp_case[i]] for i in range(len(idx_event))]
            for j in range(self._n_control)
        ]
        controls_list = [
            torch.stack([s_pred[r, c] for r, c in ctrl_fmt[j]])
            for j in range(self._n_control)
        ]
        g_control = tt.TupleTree(controls_list[0].unsqueeze(0)) if self._n_control == 1 \
            else tt.TupleTree(tuple(controls_list))

        return g_case, g_control

    @staticmethod
    def _get_timepoint_cols(df: pd.DataFrame) -> list:
        out = []
        for c in df.columns:
            if c in EPNModel._MGMT_COLS or "feat_" in c:
                continue
            try:
                float(c)
                out.append(c)
            except (ValueError, TypeError):
                pass
        return sorted(out, key=float)


# ---------------------------------------------------------------------------
# Lightning wrapper
# ---------------------------------------------------------------------------

class EPNModule(pl.LightningModule):
    """PyTorch Lightning wrapper for EPNModel."""

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

        self.epn = EPNModel(
            input_size=n_feats,
            prop_neighbors=prop_neighbors,
            alpha=alpha,
            beta=beta,
            n_control=n_control,
        )

    def training_step(self, batch: pd.DataFrame, batch_idx: int) -> torch.Tensor:
        corrected = self.epn(batch)
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
            self.epn.parameters(), lr=self._lr, weight_decay=self._wd
        )
