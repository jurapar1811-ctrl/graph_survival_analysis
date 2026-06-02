#!/usr/bin/env python3
"""
Train EPN on top of a frozen MLP for survival analysis on METABRIC.

Pipeline:
  1. Train MLP (CoxPH loss)         -- same hyper-params as run_mlp.py
  2. Freeze MLP weights
  3. Build EPNDataModule             -- pre-computes MLP survival curves
  4. Train EPNModule                 -- attention-based correction
  5. Evaluate on validation split    -- C-index + IBS

Usage:
    cd "Vanlentin's pipeline"
    python -m survival_analysis.experiments.run_epn
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as pl
from pycox.evaluation import EvalSurv

from survival_analysis.data.metabric_datamodule import MetabricGraphSurvivalDataModule
from survival_analysis.data.epn_datamodule import EPNDataModule
from survival_analysis.models.mlp_module import MLPModule
from survival_analysis.models.epn_module import EPNModule

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "metabric" / "splits.json"

# MLP hyper-params (identical to run_mlp.py)
MLP_HIDDEN_DIM = 64
MLP_LR = 0.001
MLP_WD = 5e-4
MLP_EPOCHS = 110

# EPN hyper-params
EPN_LR = 1e-3
EPN_WD = 5e-4
EPN_EPOCHS = 50
EPN_PROP_NEIGHBORS = 0.1  # fraction of patients used as neighbours per query


def evaluate_epn(epn_module: EPNModule, epn_dm: EPNDataModule) -> tuple[float, float]:
    """Evaluate corrected survival curves on the validation patients."""
    epn_module.eval()
    with torch.no_grad():
        corrected = epn_module.epn(epn_dm.val_df)  # [n_val, T]

    corrected_np = np.clip(corrected.detach().cpu().numpy(), 1e-6, 1.0)

    computed_mask = epn_dm.val_df.eval("pat_to_compute and batch").values
    val_labels = epn_dm.val_df[computed_mask][["label_duration", "label_event"]].values
    durations = val_labels[:, 0]
    events = val_labels[:, 1]

    # EvalSurv expects DataFrame [T x n_val]
    surv_df = pd.DataFrame(corrected_np.T, index=epn_dm.timepoints)
    time_grid = np.linspace(durations.min(), durations.max(), 100)

    ev = EvalSurv(surv_df, durations, events, censor_surv="km")
    cindex = ev.concordance_td()
    ibs = ev.integrated_brier_score(time_grid)
    return cindex, ibs


def run_single_split(split_index: int, n_splits: int) -> tuple[float, float]:
    print(f"\n{'='*55}")
    print(f"  Running split {split_index + 1} / {n_splits} ...")
    print(f"{'='*55}")

    # --- 1. Train MLP ---
    base_dm = MetabricGraphSurvivalDataModule(
        json_splits_path=str(SPLITS_JSON),
        split_index=split_index,
    )
    base_dm.prepare_data()
    base_dm.setup()

    mlp = MLPModule(
        in_dim=base_dm.in_dim,
        hidden_dim=MLP_HIDDEN_DIM,
        optimizer=partial(torch.optim.Adam, lr=MLP_LR, weight_decay=MLP_WD),
    )
    pl.Trainer(
        max_epochs=MLP_EPOCHS,
        accelerator="cpu",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    ).fit(model=mlp, datamodule=base_dm)
    mlp.freeze()
    mlp.eval()
    print("  MLP training complete. Weights frozen.")

    # --- 2. Build EPN DataModule ---
    epn_dm = EPNDataModule(base_datamodule=base_dm, mlp_module=mlp)
    epn_dm.setup()
    print(f"  EPN DataModule ready. "
          f"Time points: {len(epn_dm.timepoints)}, Features: {epn_dm.n_feats}")

    # --- 3. Train EPN ---
    epn = EPNModule(
        n_feats=epn_dm.n_feats,
        timepoints=epn_dm.timepoints,
        learning_rate=EPN_LR,
        weight_decay=EPN_WD,
        prop_neighbors=EPN_PROP_NEIGHBORS,
    )
    pl.Trainer(
        max_epochs=EPN_EPOCHS,
        accelerator="cpu",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    ).fit(model=epn, datamodule=epn_dm)

    # --- 4. Evaluate ---
    cindex, ibs = evaluate_epn(epn, epn_dm)
    print(f"    C-index : {cindex:.4f}")
    print(f"    IBS     : {ibs:.4f}")
    return cindex, ibs


def main() -> None:
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    n_splits = len(splits)
    print(f"\nStarting EPN experiment over {n_splits} splits\n")

    all_cindices: list[float] = []
    all_briers: list[float] = []
    rows: list[dict] = []

    for split_index in range(n_splits):
        cindex, ibs = run_single_split(split_index, n_splits)
        all_cindices.append(cindex)
        all_briers.append(ibs)
        rows.append({
            "split_index": split_index,
            "c_index": round(cindex, 4),
            "ibs": round(ibs, 4),
        })

    print(f"\n{'='*55}")
    print("  EPN — Summary across all splits")
    print(f"{'='*55}")
    print(f"  Mean C-index : {np.mean(all_cindices):.4f} ± {np.std(all_cindices):.4f}")
    print(f"  Mean IBS     : {np.mean(all_briers):.4f} ± {np.std(all_briers):.4f}")
    print(f"{'='*55}\n")

    results_df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {"split_index": "mean",
         "c_index": round(np.mean(all_cindices), 4),
         "ibs": round(np.mean(all_briers), 4)},
        {"split_index": "std",
         "c_index": round(np.std(all_cindices), 4),
         "ibs": round(np.std(all_briers), 4)},
    ])
    results_df = pd.concat([results_df, summary], ignore_index=True)

    output_path = REPO_ROOT / "results_epn_all_splits.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
