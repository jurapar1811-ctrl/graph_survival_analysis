#!/usr/bin/env python3
"""
Visualize patient similarity graph and survival curves.

Outputs (saved to repo root):
    patient_graph.png   — k-NN patient similarity graph (PCA projection)
    survival_curves.png — MLP baseline vs EPN-corrected survival curves

Usage:
    cd "Vanlentin's pipeline"
    python -m survival_analysis.experiments.visualize
"""

import sys
from functools import partial
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import lightning as pl
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from survival_analysis.data.metabric_datamodule import MetabricGraphSurvivalDataModule
from survival_analysis.data.epn_datamodule import EPNDataModule
from survival_analysis.data.graph_builder import build_patient_similarity_graph
from survival_analysis.models.mlp_module import MLPModule
from survival_analysis.models.epn_module import EPNModule
from pycox.models import CoxPH

REPO_ROOT = Path(__file__).parent.parent.parent.parent
SPLITS_JSON = REPO_ROOT / "data" / "metabric" / "splits.json"
GRAPH_K = 10


# ---------------------------------------------------------------------------
# Figure 1: Patient similarity graph (PCA 2D projection)
# ---------------------------------------------------------------------------

def plot_patient_graph(base_dm):
    """PCAで患者を2次元に投影し、k-NNエッジを描画する。"""
    print("  患者類似グラフを構築中...")

    train_x = base_dm.train_graph.x           # [N_train, F]
    train_y = base_dm.train_graph.y.numpy()   # [N_train, 2]
    events  = train_y[:, 1].astype(int)       # 0=打ち切り, 1=イベントあり

    # k-NNグラフ構築
    graph = build_patient_similarity_graph(train_x, k=GRAPH_K, metric="cosine")
    edge_index = graph.edge_index.numpy()     # [2, N*k]
    features   = train_x.numpy()             # [N, F]

    # PCAで2次元に圧縮
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(features)     # [N, 2]
    var_ratio = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(10, 8))

    # エッジを描画（薄いグレー）
    src, tgt = edge_index[0], edge_index[1]
    for s, t in zip(src, tgt):
        ax.plot(
            [coords[s, 0], coords[t, 0]],
            [coords[s, 1], coords[t, 1]],
            color="gray", alpha=0.15, linewidth=0.5, zorder=1,
        )

    # 患者ノードを描画（色分け: イベントあり=オレンジ、打ち切り=水色）
    ax.scatter(
        coords[events == 0, 0], coords[events == 0, 1],
        c="#3182bd", s=18, alpha=0.7, label=f"打ち切り ({(events==0).sum()}人)",
        zorder=2, edgecolors="white", linewidths=0.3,
    )
    ax.scatter(
        coords[events == 1, 0], coords[events == 1, 1],
        c="#e6550d", s=18, alpha=0.7, label=f"イベントあり ({(events==1).sum()}人)",
        zorder=2, edgecolors="white", linewidths=0.3,
    )

    ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)", fontsize=12)
    ax.set_title(
        f"患者類似グラフ（k-NN, k={GRAPH_K}, コサイン類似度）\n"
        f"PCA 2次元投影 — 訓練患者 {len(features)} 人",
        fontsize=13,
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = REPO_ROOT / "patient_graph.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out}")


# ---------------------------------------------------------------------------
# Figure 2: Survival curves (MLP vs EPN-corrected)
# ---------------------------------------------------------------------------

def plot_survival_curves(base_dm, epn_dm, epn_module):
    """検証患者4人の生存曲線（MLPベースライン vs EPN補正後）を描画する。"""
    print("  生存曲線を生成中...")

    mlp_module = epn_dm._mlp
    with torch.no_grad():
        cox = CoxPH(mlp_module)
        cox.compute_baseline_hazards(
            base_dm.train_graph.x,
            (base_dm.train_graph.y[..., 0], base_dm.train_graph.y[..., 1]),
        )
        mlp_surv_df = cox.predict_surv_df(base_dm.val_graph.x)  # [T × N_all]

    epn_module.eval()
    with torch.no_grad():
        epn_corr = np.clip(
            epn_module.epn(epn_dm.val_df, epn_dm.val_nn_idx).detach().numpy(),
            0, 1,
        )

    val_idx    = base_dm.val_graph.val_idx.numpy()
    val_y      = base_dm.val_graph.y.numpy()[val_idx]
    timepoints = np.array(epn_dm.timepoints)
    mlp_times  = mlp_surv_df.index.values

    # イベントあり2人・打ち切り2人を選ぶ
    event_idx  = np.where(val_y[:, 1] == 1)[0][:2]
    censor_idx = np.where(val_y[:, 1] == 0)[0][:2]
    chosen = np.concatenate([event_idx, censor_idx])
    titles = [
        "患者1（イベントあり）", "患者2（イベントあり）",
        "患者3（打ち切り）",     "患者4（打ち切り）",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, i, title in zip(axes, chosen, titles):
        pat_global = val_idx[i]
        t_event    = val_y[i, 0]
        is_event   = int(val_y[i, 1])

        ax.plot(mlp_times, mlp_surv_df.iloc[:, pat_global].values,
                color="#74c476", linewidth=2.2, label="MLP 予測")
        ax.plot(timepoints, epn_corr[i],
                color="#fd8d3c", linewidth=2.2, linestyle="--", label="EPN 補正後")
        ax.axvline(t_event, color="gray", linestyle=":", linewidth=1.2)
        ax.text(t_event + 4, 0.88,
                f"{'イベント' if is_event else '打ち切り'}\n({t_event:.0f}ヶ月)",
                fontsize=9, color="gray")
        ax.set_xlim(0, max(mlp_times[-1], timepoints[-1]) * 1.05)
        ax.set_ylim(-0.05, 1.08)
        ax.set_xlabel("時間（ヶ月）", fontsize=10)
        ax.set_ylabel("生存確率", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "生存曲線：MLP ベースライン vs EPN 補正後\n（スプリット1・検証患者の例）",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = REPO_ROOT / "survival_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    base_dm = MetabricGraphSurvivalDataModule(
        json_splits_path=str(SPLITS_JSON), split_index=0
    )
    base_dm.prepare_data()
    base_dm.setup()

    # グラフ① 患者類似グラフ（モデル訓練不要）
    print("グラフ① 患者類似グラフ ...")
    plot_patient_graph(base_dm)

    # MLP + EPN を短縮訓練（生存曲線用）
    print("\nモデルを訓練中（MLP 40ep + EPN 10ep）...")
    mlp = MLPModule(
        in_dim=base_dm.in_dim,
        hidden_dim=64,
        optimizer=partial(torch.optim.Adam, lr=0.001, weight_decay=5e-4),
    )
    pl.Trainer(
        max_epochs=40, accelerator="cpu",
        enable_progress_bar=True, enable_model_summary=False, logger=False,
    ).fit(model=mlp, datamodule=base_dm)
    mlp.freeze(); mlp.eval()

    epn_dm = EPNDataModule(base_datamodule=base_dm, mlp_module=mlp, graph_k=GRAPH_K)
    epn_dm.setup()

    epn = EPNModule(n_feats=epn_dm.n_feats, timepoints=epn_dm.timepoints)
    pl.Trainer(
        max_epochs=10, accelerator="cpu",
        enable_progress_bar=True, enable_model_summary=False, logger=False,
    ).fit(model=epn, datamodule=epn_dm)

    # グラフ② 生存曲線
    print("\nグラフ② 生存曲線 ...")
    plot_survival_curves(base_dm, epn_dm, epn)

    print("\n=== 完了 ===")
    print(f"  patient_graph.png   → {REPO_ROOT / 'patient_graph.png'}")
    print(f"  survival_curves.png → {REPO_ROOT / 'survival_curves.png'}")
