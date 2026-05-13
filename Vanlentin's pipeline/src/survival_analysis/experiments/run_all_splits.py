#!/usr/bin/env python3
"""
全スプリットを実行して平均 C-index と IBS を計算するラッパースクリプト。

実行方法:
    cd "Vanlentin's pipeline"
    python -m survival_analysis.experiments.run_all_splits
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from functools import partial
from pycox.models import CoxPH
import lightning as pl

from survival_analysis.data.metabric_datamodule import MetabricGraphSurvivalDataModule
from survival_analysis.models.dgm_model import SurvivalDGM
from survival_analysis.experiments.evaluate_dgm import evaluate

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "metabric" / "splits.json"

HIDDEN_DIM = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
TAU = 0.05
MAX_EPOCHS = 110
NB_EVAL_TESTS = 100


def run_single_split(split_index: int, n_splits: int) -> tuple[float, float, float]:
    """1つのスプリットで訓練・評価を実行して (C-index, IBS, C-index標準偏差) を返す。"""
    print(f"\n{'='*55}")
    print(f"  スプリット {split_index + 1} / {n_splits} を実行中...")
    print(f"{'='*55}")

    datamodule = MetabricGraphSurvivalDataModule(
        json_splits_path=str(SPLITS_JSON),
        split_index=split_index,
    )
    datamodule.prepare_data()
    datamodule.setup()

    optimizer_fn = partial(
        torch.optim.Adam,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    model = SurvivalDGM(
        in_dim=datamodule.in_dim,
        hid_dim=HIDDEN_DIM,
        optimizer=optimizer_fn,
        tau=TAU,
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        enable_progress_bar=True,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    survival_model = CoxPH(model)
    mean_cindex, mean_brier, std_cindex = evaluate(
        datamodule, survival_model, nb_tests=NB_EVAL_TESTS
    )

    print(f"    C-index : {mean_cindex:.4f} ± {std_cindex:.4f}")
    print(f"    IBS     : {mean_brier:.4f}")
    return mean_cindex, mean_brier, std_cindex


def main() -> None:
    with open(SPLITS_JSON) as f:
        splits = json.load(f)
    n_splits = len(splits)
    print(f"\n{n_splits} スプリットの実験を開始します（合計 {n_splits} 回訓練）\n")

    all_cindices: list[float] = []
    all_briers: list[float] = []
    all_stds: list[float] = []
    rows: list[dict] = []

    for split_index in range(n_splits):
        mean_cindex, mean_brier, std_cindex = run_single_split(split_index, n_splits)
        all_cindices.append(mean_cindex)
        all_briers.append(mean_brier)
        all_stds.append(std_cindex)
        rows.append({
            "split_index": split_index,
            "c_index": round(mean_cindex, 4),
            "ibs": round(mean_brier, 4),
            "c_index_std_within_split": round(std_cindex, 4),
        })

    print(f"\n{'='*55}")
    print("  全スプリットの結果まとめ")
    print(f"{'='*55}")
    print(f"  平均  C-index : {np.mean(all_cindices):.4f} ± {np.std(all_cindices):.4f}")
    print(f"  平均  IBS     : {np.mean(all_briers):.4f} ± {np.std(all_briers):.4f}")
    print(f"{'='*55}\n")

    results_df = pd.DataFrame(rows)
    summary = pd.DataFrame([
        {
            "split_index": "全体平均",
            "c_index": round(np.mean(all_cindices), 4),
            "ibs": round(np.mean(all_briers), 4),
            "c_index_std_within_split": round(np.mean(all_stds), 4),
        },
        {
            "split_index": "標準偏差",
            "c_index": round(np.std(all_cindices), 4),
            "ibs": round(np.std(all_briers), 4),
            "c_index_std_within_split": round(np.std(all_stds), 4),
        },
    ])
    results_df = pd.concat([results_df, summary], ignore_index=True)

    output_path = REPO_ROOT / "results_all_splits.csv"
    results_df.to_csv(output_path, index=False)
    print(f"結果を保存しました: {output_path}")


if __name__ == "__main__":
    main()
