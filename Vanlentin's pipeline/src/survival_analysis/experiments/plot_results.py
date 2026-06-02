#!/usr/bin/env python3
"""
Visualize experiment results.

Generates three figures saved to the repo root:
  1. results_comparison.png  — C-index / IBS comparison across all models
  2. results_per_split.png   — per-split C-index for each model
  3. survival_curves.png     — MLP baseline vs EPN-corrected curves for example patients

Usage:
    cd "Vanlentin's pipeline"
    python -m survival_analysis.experiments.plot_results
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from survival_analysis.data.metabric_datamodule import MetabricGraphSurvivalDataModule
from survival_analysis.data.epn_datamodule import EPNDataModule
from survival_analysis.models.mlp_module import MLPModule
from survival_analysis.models.epn_module import EPNModule

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "metabric" / "splits.json"
OUT_DIR = REPO_ROOT

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "SurvivalDGM": "#6baed6",
    "MLP": "#74c476",
    "EPN (グラフあり)": "#fd8d3c",
}


# ---------------------------------------------------------------------------
# Figure 1: Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison():
    dgm = pd.read_csv(REPO_ROOT / "results_all_splits.csv")
    mlp = pd.read_csv(REPO_ROOT / "results_mlp_all_splits.csv")
    epn = pd.read_csv(REPO_ROOT / "results_epn_all_splits.csv")

    def get_mean_std(df, col):
        mean = float(df[df["split_index"] == "mean"][col].iloc[0])
        std  = float(df[df["split_index"] == "std"][col].iloc[0])
        return mean, std

    models = ["SurvivalDGM", "MLP", "EPN (グラフあり)"]
    c_means, c_stds   = zip(*[get_mean_std(d, "c_index") for d in [dgm, mlp, epn]])
    ibs_means, ibs_stds = zip(*[get_mean_std(d, "ibs")   for d in [dgm, mlp, epn]])
    colors = [COLORS[m] for m in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, means, stds, title, ylabel, ylim, better in [
        (axes[0], c_means,   c_stds,   "C-index（高いほど良い）",  "C-index", (0.55, 0.70), "↑"),
        (axes[1], ibs_means, ibs_stds, "IBS（低いほど良い）", "Integrated Brier Score", (0.14, 0.20), "↓"),
    ]:
        bars = ax.bar(x, means, 0.5, yerr=stds, capsize=6,
                      color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.003 if better == "↑" else 0.001),
                    f"{mean:.4f}", ha="center", va="bottom", fontsize=10)

    fig.suptitle("METABRIC データセット — 5スプリット評価結果（平均 ± 標準偏差）",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "results_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Per-split C-index
# ---------------------------------------------------------------------------

def plot_per_split():
    dgm = pd.read_csv(REPO_ROOT / "results_all_splits.csv")
    mlp = pd.read_csv(REPO_ROOT / "results_mlp_all_splits.csv")
    epn = pd.read_csv(REPO_ROOT / "results_epn_all_splits.csv")

    def splits_only(df):
        return df[df["split_index"].apply(lambda x: str(x).isdigit())]["c_index"].astype(float).tolist()

    splits = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(splits, splits_only(dgm), "o-", color=COLORS["SurvivalDGM"],
            linewidth=2, markersize=8, label="SurvivalDGM")
    ax.plot(splits, splits_only(mlp), "s-", color=COLORS["MLP"],
            linewidth=2, markersize=8, label="MLP")
    ax.plot(splits, splits_only(epn), "^-", color=COLORS["EPN (グラフあり)"],
            linewidth=2, markersize=8, label="EPN (グラフあり)")
    ax.set_xlabel("スプリット番号", fontsize=12)
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_title("スプリットごとの C-index", fontsize=14)
    ax.set_xticks(splits)
    ax.set_ylim(0.55, 0.70)
    ax.legend(fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / "results_per_split.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out}")


# ---------------------------------------------------------------------------
# Figure 3: Survival curves (MLP baseline vs EPN-corrected)
# ---------------------------------------------------------------------------

def plot_survival_curves():
    import lightning as pl
    from pycox.models import CoxPH

    print("  モデルを短縮訓練中（30ep MLP + 10ep EPN）...")
    base_dm = MetabricGraphSurvivalDataModule(
        json_splits_path=str(SPLITS_JSON), split_index=0
    )
    base_dm.prepare_data(); base_dm.setup()

    mlp = MLPModule(
        in_dim=base_dm.in_dim, hidden_dim=64,
        optimizer=partial(torch.optim.Adam, lr=0.001, weight_decay=5e-4),
    )
    pl.Trainer(max_epochs=30, accelerator="cpu",
               enable_progress_bar=False, enable_model_summary=False,
               logger=False).fit(model=mlp, datamodule=base_dm)
    mlp.freeze(); mlp.eval()

    epn_dm = EPNDataModule(base_datamodule=base_dm, mlp_module=mlp, graph_k=10)
    epn_dm.setup()
    epn = EPNModule(n_feats=epn_dm.n_feats, timepoints=epn_dm.timepoints)
    pl.Trainer(max_epochs=10, accelerator="cpu",
               enable_progress_bar=False, enable_model_summary=False,
               logger=False).fit(model=epn, datamodule=epn_dm)
    epn.eval()

    # MLP baseline survival curves for all patients
    with torch.no_grad():
        cox = CoxPH(mlp)
        cox.compute_baseline_hazards(
            base_dm.train_graph.x,
            (base_dm.train_graph.y[..., 0], base_dm.train_graph.y[..., 1])
        )
        mlp_surv_df = cox.predict_surv_df(base_dm.val_graph.x)   # [T × N_all]

    val_idx  = base_dm.val_graph.val_idx.numpy()
    val_y    = base_dm.val_graph.y.numpy()
    val_y_v  = val_y[val_idx]

    # EPN-corrected curves for val patients
    with torch.no_grad():
        epn_corr = np.clip(
            epn.epn(epn_dm.val_df, epn_dm.val_nn_idx).detach().numpy(), 0, 1
        )
    timepoints = np.array(epn_dm.timepoints)
    mlp_times  = mlp_surv_df.index.values

    # Pick 4 example patients
    event_idx  = np.where(val_y_v[:, 1] == 1)[0][:2]
    censor_idx = np.where(val_y_v[:, 1] == 0)[0][:2]
    chosen = np.concatenate([event_idx, censor_idx])
    titles = ["患者1（イベントあり）", "患者2（イベントあり）",
              "患者3（打ち切り）",     "患者4（打ち切り）"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, i, title in zip(axes, chosen, titles):
        pat_global = val_idx[i]
        t_event    = val_y_v[i, 0]
        is_event   = int(val_y_v[i, 1])

        mlp_curve = mlp_surv_df.iloc[:, pat_global].values
        epn_curve = epn_corr[i]

        ax.plot(mlp_times, mlp_curve, color="#74c476", linewidth=2.2,
                label="MLP 予測")
        ax.plot(timepoints, epn_curve, color="#fd8d3c", linewidth=2.2,
                linestyle="--", label="EPN 補正後")
        ax.axvline(t_event, color="gray", linestyle=":", linewidth=1.2)
        label_txt = f"{'イベント' if is_event else '打ち切り'}\n({t_event:.0f}ヶ月)"
        ax.text(t_event + 3, 0.88, label_txt, fontsize=9, color="gray")
        ax.set_xlim(0, max(mlp_times[-1], timepoints[-1]) * 1.05)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("時間（ヶ月）", fontsize=10)
        ax.set_ylabel("生存確率", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)

    fig.suptitle("生存曲線：MLP ベースライン vs EPN 補正後\n（スプリット1 検証患者の例）",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "survival_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out}")


if __name__ == "__main__":
    print("グラフ① モデル比較棒グラフ ...")
    plot_comparison()
    print("グラフ② スプリット別 C-index ...")
    plot_per_split()
    print("グラフ③ 生存曲線の可視化 ...")
    plot_survival_curves()
    print("\n=== 完了 ===")
