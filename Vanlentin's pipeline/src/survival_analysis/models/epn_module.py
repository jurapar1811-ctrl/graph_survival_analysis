#!/usr/bin/env python3
"""
EPNModule: Graph-based Error Propagation Network as a PyTorch Lightning module.

Architecture
------------
- A k-NN patient similarity graph (built by EPNDataModule using graph_builder.py)
  determines *which* patients are neighbours for each query patient.
- A learned attention mechanism (key/query projections) determines *how much*
  weight to give each neighbour's prediction error.
- The weighted errors are added to the MLP baseline prediction as a correction.

No imports from surv_epn — all logic is self-contained here.
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
# Standalone helpers
# ---------------------------------------------------------------------------

def _get_hazard_from_survival_curve(pred: torch.Tensor) -> torch.Tensor:
    """Convert survival curve [N, T] to discrete hazard [N, T]."""
    cum_hazard = -torch.log(torch.clamp(pred, min=1e-9))
    return torch.diff(cum_hazard, prepend=torch.zeros(pred.shape[0], 1))


# ---------------------------------------------------------------------------
# Core EPN model
# ---------------------------------------------------------------------------

class EPNModel(nn.Module):
    """
    Graph-based Error Propagation Network.

    forward() inputs
    ----------------
    df      : pd.DataFrame  — patient data (features, MLP survival curves, mgmt cols)
    nn_idx  : torch.Tensor [N, k]  — precomputed k-NN neighbour indices from graph_builder

    forward() output
    ----------------
    torch.Tensor [n_batch, T] — graph-corrected survival curves for the batch patients
    """

    _MGMT_COLS = {"label_duration", "label_event", "split",
                  "pat_to_compute", "batch", "pat_id", "index"}

    def __init__(
        self,
        input_size: int,
        temp_att: float = 1.0,
        alpha: float = 0.0,
        beta: float = 0.0,
        n_control: int = 1,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        self._temperature = temp_att
        self._alpha = alpha
        self._beta = beta
        self._n_control = n_control
        self._dk = torch.sqrt(torch.tensor(float(input_size)))

        # Learnable projections for attention scoring
        self._key_projection = nn.Linear(input_size, input_size)
        self._query_projection = nn.Linear(input_size, input_size)
        self._criterion = CoxCCLoss(shrink=0.0)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, df: pd.DataFrame, nn_idx: torch.Tensor) -> torch.Tensor:
        """
        Graph-corrected survival curves for batch patients.

        For each batch patient i:
          1. Look up its k neighbours from nn_idx[i].
          2. Compute attention scores to those k neighbours (learned projections).
          3. Mask self-attention and (in val mode) test-to-test attention.
          4. Correction = softmax(att) · errors[neighbours].
          5. Return MLP_pred[i] + correction[i].

        Memory: [n_batch, k, T]  (vs [n_batch, n_batch, T] in the attention-only approach)
        """
        tp_cols = self._get_timepoint_cols(df)
        T = len(tp_cols)
        preds = torch.from_numpy(df[tp_cols].values).float()            # [N, T]
        labels = torch.from_numpy(
            df[["label_duration", "label_event"]].values
        ).float()                                                         # [N, 2]
        feat_cols = [c for c in df.columns if "feat_" in c]
        features = torch.from_numpy(df[feat_cols].values).float()       # [N, F]
        F_dim = features.shape[1]

        test_idx = np.where(df.eval("pat_to_compute and batch"))[0]     # [n_batch]
        n_batch = len(test_idx)
        training = (
            len(np.where(df.eval(
                'pat_to_compute and (split=="val" or split=="test")'
            ))[0]) == 0
        )

        # Errors for all N patients: [N, T]
        timepoints = torch.tensor([float(c) for c in tp_cols])
        errors = self._compute_error_survival(timepoints, preds, labels)

        # Neighbours for batch patients: [n_batch, k]
        test_idx_t = torch.tensor(test_idx)
        neighbour_idx = nn_idx[test_idx_t]                               # [n_batch, k]
        k = neighbour_idx.shape[1]

        # --- Graph-constrained attention ---
        # Query: projected features of test patients        [n_batch, F]
        # Key:   projected features of their k neighbours  [n_batch, k, F]
        q = self._query_projection(features[test_idx_t])                 # [n_batch, F]
        neighbour_feats = features[neighbour_idx]                        # [n_batch, k, F]
        keys = self._key_projection(
            neighbour_feats.view(-1, F_dim)
        ).view(n_batch, k, F_dim)                                        # [n_batch, k, F]

        # Scaled dot-product: [n_batch, 1, F] × [n_batch, F, k] → [n_batch, k]
        att = torch.bmm(q.unsqueeze(1), keys.permute(0, 2, 1)).squeeze(1) / self._dk

        # --- Masking ---
        if not training:
            # Val mode: prevent val patients from attending to other val patients
            val_patient_idx = torch.tensor(
                np.where(df["split"].values == "val")[0]
            )
            val_mask = torch.isin(neighbour_idx, val_patient_idx)        # [n_batch, k]
            att = att.masked_fill(val_mask, torch.finfo(torch.float32).min)
        else:
            # Train mode: zero out self-attention
            self_mask = (neighbour_idx == test_idx_t.unsqueeze(1))       # [n_batch, k]
            att = att.masked_fill(self_mask, torch.finfo(torch.float32).min)

        att = F.softmax(att * self._temperature, dim=-1)                 # [n_batch, k]

        # Weighted correction: [n_batch, k, T] → [n_batch, T]
        neighbour_errors = errors[neighbour_idx]                         # [n_batch, k, T]
        correction = torch.einsum("bk,bkt->bt", att, neighbour_errors)

        return preds[test_idx_t] + correction                            # [n_batch, T]

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(
        self,
        timepoints: torch.Tensor,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """CoxCC loss with optional elastic-net penalty."""
        l1, l2 = torch.tensor(0.0), torch.tensor(0.0)
        for _, w in self.named_parameters():
            l1 = l1 + w.abs().sum()
            l2 = l2 + w.pow(2).sum()

        pred_hazard = _get_hazard_from_survival_curve(pred)
        g_case, g_control = self._make_case_control(timepoints, pred_hazard, y)
        return self._criterion(g_case, g_control) + self._alpha * l1 + self._beta * l2

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_error_survival(
        timepoints: torch.Tensor,
        pred: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Signed error between predicted survival curve and step-label [N, T]."""
        T, N = timepoints.shape[0], y.shape[0]
        cont = torch.where(
            timepoints.repeat(N, 1) > y[:, 0].repeat(T, 1).t(),
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
    def _sorted_input_target(pred: torch.Tensor, y: torch.Tensor):
        durations_np = y[:, 0].detach().cpu().numpy()
        idx_sort = np.argsort(durations_np)
        if (idx_sort == np.arange(len(idx_sort))).all():
            return pred, y
        idx_t = torch.tensor(idx_sort)
        return pred[idx_t], y[idx_t]

    @staticmethod
    def _make_at_risk_dict(durations: np.ndarray) -> dict:
        import pandas as _pd
        s = _pd.Series(durations)
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
        s_pred, s_y = self._sorted_input_target(pred_hazard, y)
        at_risk = self._make_at_risk_dict(s_y[:, 0].detach().cpu().numpy())
        tp_np = timepoints.detach().cpu().numpy()

        idx_event = torch.where(s_y[:, 1] == 1.0)[0]
        tp_case = np.where(
            [tp_np == s_y[i, 0].item() for i in idx_event]
        )[1]

        idx_case = [[idx_event[i].item(), tp_case[i]] for i in range(len(idx_event))]
        g_case = torch.stack([s_pred[r, c] for r, c in idx_case])

        ctrl_idx = self._sample_alive_from_dates(
            s_y[idx_event, 0].detach().cpu().numpy(), at_risk, self._n_control
        )
        ctrl_fmt = [
            [[ctrl_idx[i, j], tp_case[i]] for i in range(len(idx_event))]
            for j in range(self._n_control)
        ]
        controls_list = [
            torch.stack([s_pred[r, c] for r, c in ctrl_fmt[j]])
            for j in range(self._n_control)
        ]
        g_control = (
            tt.TupleTree(controls_list[0].unsqueeze(0))
            if self._n_control == 1
            else tt.TupleTree(tuple(controls_list))
        )
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
    """PyTorch Lightning wrapper for graph-based EPNModel.

    Each batch from EPNDataModule is a dict {"df": ..., "nn_idx": ...}.
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["timepoints"])
        self._timepoints = torch.tensor(timepoints, dtype=torch.float32)
        self._lr = learning_rate
        self._wd = weight_decay

        self.epn = EPNModel(
            input_size=n_feats,
            alpha=alpha,
            beta=beta,
            n_control=n_control,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        df, nn_idx = batch["df"], batch["nn_idx"]
        corrected = self.epn(df, nn_idx)

        computed_mask = df.eval("pat_to_compute and batch").values
        labels = torch.tensor(
            df[computed_mask][["label_duration", "label_event"]].values,
            dtype=torch.float32,
        )
        loss = self.epn.loss(self._timepoints, corrected, labels)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        df, nn_idx = batch["df"], batch["nn_idx"]
        with torch.no_grad():
            corrected = self.epn(df, nn_idx)
        self.log("val/n_preds", float(corrected.shape[0]),
                 on_step=False, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.epn.parameters(), lr=self._lr, weight_decay=self._wd
        )
