#!/usr/bin/env python3
"""
run_epn_v2.py — clean entry script for MLP + graph-EPN pipeline.

Structure mirrors run_mlp.py / run_all_splits.py (per supervisor request).
Trains MLP first, then EPN on top, and reports a side-by-side comparison.

Usage:
    cd "Vanlentin's pipeline"
    python -m survival_analysis.experiments.run_epn_v2
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as pl
from pycox.evaluation import EvalSurv
from pycox.models import CoxPH

from survival_analysis.data.metabric_datamodule import MetabricGraphSurvivalDataModule
from survival_analysis.data.epn_datamodule import EPNDataModule
from survival_analysis.models.mlp_module import MLPModule
from survival_analysis.models.epn_module import EPNModule
from survival_analysis.experiments.evaluate_dgm import evaluate

REPO_ROOT   = Path(__file__).parent.parent.parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "metabric" / "splits.json"

# ── Hyper-parameters ────────────────────────────────────────────────────────
MLP_HIDDEN_DIM = 64
MLP_LR         = 1e-3
MLP_WD         = 5e-4
MLP_EPOCHS     = 110

EPN_LR         = 1e-3
EPN_WD         = 5e-4
EPN_EPOCHS     = 50
GRAPH_K        = 10
GRAPH_METRIC   = "cosine"
NB_EVAL_MLP    = 10


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_mlp(datamodule, mlp: MLPModule) -> tuple:
    """Wrap MLP with PyCox CoxPH and reuse the existing evaluate() helper."""
    survival_model = CoxPH(mlp)
    mean_c, mean_ibs, _ = evaluate(datamodule, survival_model, nb_tests=NB_EVAL_MLP)
    return mean_c, mean_ibs


def evaluate_epn(epn: EPNModule, dm: EPNDataModule) -> tuple:
    """Evaluate graph-EPN on validation patients using EvalSurv."""
    epn.eval()
    with torch.no_grad():
        b = next(iter(dm.val_dataloader()))
        corrected = epn.epn(
            b["features"], b["surv_preds"], b["labels"],
            b["nn_idx"],   b["test_idx"],   b["val_patient_idx"],
        )

    corrected_np = np.clip(corrected.cpu().numpy(), 1e-6, 1.0)
    val_labels   = dm.all_labels[dm.val_patient_idx].numpy()
    durations, events = val_labels[:, 0], val_labels[:, 1]

    surv_df   = pd.DataFrame(corrected_np.T, index=dm.timepoints)
    time_grid = np.linspace(durations.min(), durations.max(), 100)

    ev    = EvalSurv(surv_df, durations, events, censor_surv="km")
    return ev.concordance_td(), ev.integrated_brier_score(time_grid)


# ---------------------------------------------------------------------------
# Single-split pipeline
# ---------------------------------------------------------------------------

def run_split(split_index: int, n_splits: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  Split {split_index + 1} / {n_splits}")
    print(f"{'='*60}")

    base_dm = MetabricGraphSurvivalDataModule(
        json_splits_path=str(SPLITS_JSON), split_index=split_index
    )
    base_dm.prepare_data()
    base_dm.setup()

    # ── Train MLP ────────────────────────────────────────────────────────────
    mlp = MLPModule(
        in_dim=base_dm.in_dim,
        hidden_dim=MLP_HIDDEN_DIM,
        optimizer=partial(torch.optim.Adam, lr=MLP_LR, weight_decay=MLP_WD),
    )
    pl.Trainer(
        max_epochs=MLP_EPOCHS, accelerator="cpu",
        enable_progress_bar=True, enable_model_summary=False, logger=False,
    ).fit(model=mlp, datamodule=base_dm)

    mlp_c, mlp_ibs = evaluate_mlp(base_dm, mlp)
    print(f"  [MLP]  C-index: {mlp_c:.4f}   IBS: {mlp_ibs:.4f}")

    mlp.freeze()
    mlp.eval()

    # ── Train EPN ────────────────────────────────────────────────────────────
    epn_dm = EPNDataModule(
        base_datamodule=base_dm, mlp_module=mlp,
        graph_k=GRAPH_K, graph_metric=GRAPH_METRIC,
    )
    epn_dm.setup()

    epn = EPNModule(
        n_feats=epn_dm.n_feats,
        timepoints=epn_dm.timepoints,
        learning_rate=EPN_LR,
        weight_decay=EPN_WD,
    )
    pl.Trainer(
        max_epochs=EPN_EPOCHS, accelerator="cpu",
        enable_progress_bar=True, enable_model_summary=False, logger=False,
    ).fit(model=epn, datamodule=epn_dm)

    epn_c, epn_ibs = evaluate_epn(epn, epn_dm)
    print(f"  [EPN]  C-index: {epn_c:.4f}   IBS: {epn_ibs:.4f}")

    return {
        "split_index": split_index,
        "mlp_c_index": round(mlp_c,   4),
        "mlp_ibs":     round(mlp_ibs, 4),
        "epn_c_index": round(epn_c,   4),
        "epn_ibs":     round(epn_ibs, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with open(SPLITS_JSON) as f:
        n_splits = len(json.load(f))

    print(f"\nMLP vs Graph-EPN — {n_splits} splits")
    print(f"  MLP  : hidden={MLP_HIDDEN_DIM}, epochs={MLP_EPOCHS}")
    print(f"  EPN  : epochs={EPN_EPOCHS}, k={GRAPH_K}, metric={GRAPH_METRIC}")

    rows = [run_split(i, n_splits) for i in range(n_splits)]

    mlp_c = [r["mlp_c_index"] for r in rows]
    mlp_b = [r["mlp_ibs"]     for r in rows]
    epn_c = [r["epn_c_index"] for r in rows]
    epn_b = [r["epn_ibs"]     for r in rows]

    print(f"\n{'='*60}")
    print("  Final comparison")
    print(f"{'='*60}")
    print(f"  MLP  C-index : {np.mean(mlp_c):.4f} ± {np.std(mlp_c):.4f}")
    print(f"  MLP  IBS     : {np.mean(mlp_b):.4f} ± {np.std(mlp_b):.4f}")
    print(f"  EPN  C-index : {np.mean(epn_c):.4f} ± {np.std(epn_c):.4f}")
    print(f"  EPN  IBS     : {np.mean(epn_b):.4f} ± {np.std(epn_b):.4f}")
    print(f"{'='*60}\n")

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"split_index": "mean",
         "mlp_c_index": round(np.mean(mlp_c), 4), "mlp_ibs": round(np.mean(mlp_b), 4),
         "epn_c_index": round(np.mean(epn_c), 4), "epn_ibs": round(np.mean(epn_b), 4)},
        {"split_index": "std",
         "mlp_c_index": round(np.std(mlp_c), 4),  "mlp_ibs": round(np.std(mlp_b), 4),
         "epn_c_index": round(np.std(epn_c), 4),  "epn_ibs": round(np.std(epn_b), 4)},
    ])
    out = REPO_ROOT / "results_comparison_v2.csv"
    pd.concat([df, summary], ignore_index=True).to_csv(out, index=False)
    print(f"Results saved to: {out}")


if __name__ == "__main__":
    main()
