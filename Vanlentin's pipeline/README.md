# Graph Survival Analysis

グラフニューラルネットワーク（GNN）を用いた生存時間解析フレームワーク。Dynamic Graph Module（DGM）により患者データからグラフ構造を動的に学習し、Cox比例ハザードモデルと組み合わせて生存予測を行う。

## 概要

`SurvivalDGM` モデルは患者の特徴量からエッジ（接続関係）確率を確率的に学習する。GATv2Conv によるメッセージパッシングと Cox 部分尤度損失を組み合わせ、グラフ構造の学習と生存時間予測を同時に実施する。評価指標は C-index（一致率）と Brier score を使用し、複数回推論の平均で報告する。

## インストール

大学ネットワーク等 PyPI へのアクセスが制限されている環境では、conda で依存関係を管理する。

```bash
# 専用環境を作成
conda create -n graph_survival python=3.12 -y
conda activate graph_survival

# PyTorch と主要ライブラリ（conda-forge 経由）
conda install -c conda-forge pytorch torchvision -y
conda install -c conda-forge \
    lightning=2.5.5 hydra-core hydra-colorlog \
    plotly python-louvain scikit-survival pycox \
    scikit-learn pandas matplotlib rich scipy=1.13.1 -y

# pip 経由
python -m pip install torch-geometric wandb lifelines

# プロジェクト本体を editable install
python -m pip install -e . --no-deps
```

> **scipy=1.13.1 について:** scipy>=1.14 では pycox が使用する `scipy.integrate.simps` が削除されているため、1.13.x が必要。

## データ

### Metabric

Metabric データセットは `pycox` 経由で初回実行時に自動ダウンロードされる。

スプリットファイルはリポジトリの `data/metabric/splits.json` に配置する。

```
data/
└── metabric/
    └── splits.json
```

`splits.json` の構造：

```json
[
  { "train": [381, 382, ...], "test": [0, 1, ...] },
  { "train": [...], "test": [...] }
]
```

## 実行

```bash
uv run --no-sync python -m survival_analysis.experiments.evaluate_dgm \
    logger=csv \
    data=metabric
```

> **`--no-sync` について:** PyPI 制限環境では uv の自動パッケージ同期をスキップするためにこのフラグが必要。

### 主な Hydra オーバーライド

| 引数 | 説明 |
|------|------|
| `logger=csv` | CSV にログ保存（wandb 使用不可環境向け） |
| `logger=wandb` | Weights & Biases でログ記録（デフォルト） |
| `data=metabric` | Metabric データセットを使用 |
| `data.split_index=1` | 使用するスプリットのインデックス（デフォルト: 0） |
| `trainer=cpu` | CPU 強制使用 |
| `trainer=mps` | Apple Silicon GPU を使用 |

## プロジェクト構造

```
graph_survival_analysis/
├── configs/
│   ├── data/metabric.yaml          # Metabric データ設定
│   ├── experiment/eval_dgm.yaml    # 実験設定
│   ├── model/dgm.yaml              # SurvivalDGM モデル設定
│   └── trainer/                    # トレーナー設定群
├── data/
│   └── metabric/splits.json        # train/test スプリット
└── src/survival_analysis/
    ├── data/metabric_datamodule.py  # データ読み込み・前処理
    ├── experiments/evaluate_dgm.py  # メイン評価スクリプト
    └── models/dgm_model.py          # SurvivalDGM モデル定義
```

## モデルの仕組み

1. **特徴量エンコード** — 線形層 `phi` で入力を潜在空間へ射影
2. **グラフ推定** — 潜在表現の内積からエッジ確率 `pi` を計算（Binary Concrete でサンプリング、上三角行列を対称化して無向グラフ化）
3. **メッセージパッシング** — GATv2Conv で近傍情報を集約
4. **リスクスコア予測** — 線形層 `out` で各患者の Cox リスクスコアを出力
5. **損失関数** — Cox 部分尤度 + L1 スパース正則化 + エントロピー正則化

学習後、`evaluate()` が 100 回の推論を実施して C-index と Brier score の平均・標準偏差を算出する。
