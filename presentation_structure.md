# プレゼンテーション構成案
## Modernization and Integration of a New Graph-Based Deep Learning Method for Survival Analysis
**Yosuke Kimoto, Sena Fukabe — ECN Ecole Centrale de Nantes, June 2026**
**想定発表時間: 20分 (Q&A: 5〜10分)**

---

## スライド一覧と時間配分

| # | タイトル | 時間目安 |
|---|---------|---------|
| 1 | タイトル | 0:30 |
| 2 | アジェンダ | 0:30 |
| 3 | 背景 — 生存時間解析とは | 1:00 |
| 4 | プロジェクトの目的 | 1:00 |
| 5 | 使用モデルの概観 (3モデル) | 1:30 |
| 6 | MLP の構造 | 1:00 |
| 7 | EPN の原理 — エラー伝播ネットワーク | 2:00 |
| 8 | リポジトリ構成 (視覚化) | 1:30 |
| 9 | 2段階パイプライン全体像 | 1:30 |
| 10 | 患者類似度グラフの構築 | 1:30 |
| 11 | 技術的課題①: DataFrameの排除 | 1:00 |
| 12 | 技術的課題②: メモリ問題の解決 | 1:30 |
| 13 | Hydra による設定管理 | 0:30 |
| 14 | 結果 — 患者類似度グラフの可視化 | 1:00 |
| 15 | 結果 — 生存曲線の比較 | 1:00 |
| 16 | 結果 — 定量的評価 (C-index / IBS) | 1:30 |
| 17 | 考察 — なぜEPNはMLPを上回らないか | 1:30 |
| 18 | 今後の改善方向 | 0:30 |
| 19 | まとめ | 0:30 |
| 20 | 参考文献 | — |
| +  | バックアップスライド (Q&A用) | — |

---

## 各スライドの詳細

---

### スライド 1: タイトル

```
タイトル:
  Modernization and Integration of a New
  Graph-Based Deep Learning Method for Survival Analysis

著者: Yosuke Kimoto, Sena Fukabe
所属: ECN — Ecole Centrale de Nantes
日付: June 2026
```

---

### スライド 2: アジェンダ

```
1. 背景: 生存時間解析の概要
2. プロジェクトの目的
3. 使用モデルの説明 (MLP / EPN / SurvivalDGM)
4. パイプラインの実装
   - リポジトリ構成
   - 2段階パイプライン
   - 技術的課題と解決策
5. 実験結果
6. 考察と今後の展望
```

---

### スライド 3: 背景 — 生存時間解析とは

```
【左半分: テキスト】
■ 生存時間解析の特性
  - 目標: 患者が生存し続ける確率 S(t|x) を予測
  - 通常の回帰と異なる点: 打ち切りデータ (censored observations)
    → 観察中にイベント未発生の患者を正しく扱う必要がある

■ 評価指標
  - C-index (Concordance Index)
      リスクのランキング精度 (0.5=ランダム, 1.0=完全)
  - IBS (Integrated Brier Score)
      予測確率と実際の二値結果の差 (低いほど良い)

■ 使用データ: METABRIC
  - 乳癌患者 1,904名
  - 9個の臨床特徴量
  - イベント率 ~58% (死亡 = 1)

【右半分: 図】
  S(t|x) の概念図 (生存曲線のイメージ)
  打ち切りデータの説明図
```

---

### スライド 4: プロジェクトの目的

```
■ 主目的
  既存の Valentin の PyTorch Lightning パイプラインに
  Error Propagation Network (EPN) を統合する
  ※ コードのコピーペーストは行わない

■ 段階的な統合ステップ
  Step 1  既存パイプライン (SurvivalDGM) の動作確認 (5-fold CV)
  Step 2  Oriane の MLP を Lightning モジュールとして実装
  Step 3  PyTorch Geometric による患者類似度グラフの構築
  Step 4  EPN 補正モデルをテンソルベースで再実装
  Step 5  Hydra による設定管理の統合

■ 指導教員からの主な指摘事項
  - DataFrame依存はBad Practice → テンソルのみで実装すること
  - グラフ構築を先に独立して実装してから統合すること
```

---

### スライド 5: 使用モデルの概観 (3モデル)

```
【比較表形式のスライド】

+-----------------+--------------------+---------------------------+
| モデル          | 種別               | 特徴                      |
+-----------------+--------------------+---------------------------+
| SurvivalDGM     | 確率的グラフモデル | 既存ベースライン          |
| (Valentin)      |                    | 確率的グラフ + 100回推論  |
|                 |                    | C-index: 0.6043           |
+-----------------+--------------------+---------------------------+
| MLP             | フィードフォワード | Stage 1 モデル            |
| (Oriane)        | ニューラルネット   | CoxPH損失                 |
|                 |                    | C-index: 0.6495 ← 最高   |
+-----------------+--------------------+---------------------------+
| EPN             | グラフ注意機構     | Stage 2 モデル            |
| (This project)  |                    | MLP予測を近傍患者で補正   |
|                 |                    | C-index: 0.6157           |
+-----------------+--------------------+---------------------------+
```

---

### スライド 6: MLP の構造

```
【ネットワーク図】

Input (9特徴量)
    ↓
Linear(64) → ReLU
    ↓
Linear(64) → ReLU
    ↓
Linear(1) → リスクスコア
    ↓
PyCox CoxPH wrapper
    ↓
生存曲線 S(t|x) [N, T]

■ 学習設定
  - 損失関数: CoxPH (比例ハザード偏微分尤度)
  - エポック: 110
  - Adam optimizer, lr = 0.001
  - 評価: 5-fold CV (METABRIC)
```

---

### スライド 7: EPN の原理 — エラー伝播ネットワーク

```
【図: 患者Aの補正プロセス】

考え方: MLPが似た患者に対して同じ方向の誤差を犯すなら
        近傍患者の誤差を使って補正できる

補正の手順 (患者Aに対して):
  1. k-NNグラフで患者Aに類似した k=10 の近傍患者を特定
  2. 各近傍jのMLP予測誤差を計算:
       error_j(t) = true_survival_j(t) - MLP_pred_j(t)
  3. 注意重みを計算 (学習可能なKey/Query射影):
       att_j = softmax( (W_q · feat_A) · (W_k · feat_j) / √F )
  4. 補正量を計算:
       correction(t) = Σ_j ( att_j × error_j(t) )
  5. 補正後の予測:
       S_corrected_A(t) = MLP_pred_A(t) + correction(t)

学習: CoxCC損失 (Case-Control Cox) + ケースコントロールサンプリング
```

---

### スライド 8: リポジトリ構成 (視覚化)

```
【メインの図: ツリー図 + 色分け凡例】

Vanlentin's pipeline/
│
├── configs/                          [設定管理]
│   ├── experiment/
│   │   └── run_epn.yaml   ←── Hydra メイン設定
│   └── model/{mlp.yaml, epn.yaml}
│
├── src/survival_analysis/
│   │
│   ├── data/               [データ処理層]
│   │   ├── metabric_datamodule.py  (既存) METABRICデータ読込
│   │   ├── graph_builder.py        (★新規) k-NN患者グラフ構築
│   │   └── epn_datamodule.py       (★新規) MLP曲線+グラフ前計算
│   │
│   ├── models/             [モデル層]
│   │   ├── mlp_module.py           (★新規) MLP LightningModule
│   │   ├── epn_module.py           (★新規) EPN LightningModule
│   │   └── dgm_model.py            (既存) SurvivalDGM
│   │
│   └── experiments/        [実験スクリプト]
│       ├── run_mlp.py              (★新規) MLP 5-fold実行
│       ├── run_epn_v2.py           (★新規) MLP+EPN全体 (Hydraエントリ)
│       └── visualize.py            (★新規) グラフ・曲線可視化
│
└── data/metabric/splits.json       [5-fold CV 分割定義]

凡例: (既存) = Valentinのコード  (★新規) = このプロジェクトで追加
```

---

### スライド 9: 2段階パイプライン全体像

```
【フロー図: 上下2ブロック構成】

┌─ Stage 1: MLP 学習 ─────────────────────────────────────┐
│  MetabricGraphSurvivalDataModule (METABRIC, split i)      │
│    ↓                                                       │
│  MLPModule: 110エポック, CoxPH損失, Adam lr=0.001         │
│    ↓                                                       │
│  評価: CoxPH wrapper → C-index, IBS                       │
│    ↓                                                       │
│  mlp.freeze()  ← 重みを永続的にロック                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─ Stage 2: EPN 学習 ─────────────────────────────────────┐
│  EPNDataModule.setup():                                    │
│    ├─ 全患者のMLP生存曲線を事前計算 [N_all, T]            │
│    ├─ 訓練用k-NNグラフ → train_nn_idx [1523, 10]          │
│    └─ 検証用k-NNグラフ → val_nn_idx [1904, 10]            │
│    ↓                                                       │
│  EPNModule: 50エポック, CoxCC損失, Adam lr=0.001           │
│    ↓                                                       │
│  評価: EvalSurv → C-index, IBS                            │
└─────────────────────────────────────────────────────────┘

データ分割: 5-fold CV
  訓練: ~1,523患者 / 検証: ~381患者
```

---

### スライド 10: 患者類似度グラフの構築

```
【左: 概念図 (患者ノードと接続エッジ)】

  患者A ─── 患者B  (コサイン類似度が高い)
       \
        ── 患者C
       \
        ── ...

【右: 実装の詳細】

■ 構築手法
  - ライブラリ: PyTorch Geometric (PyG)
  - 距離指標: コサイン類似度 (スケール不変)
  - 近傍数: k = 10
  - 自己ループ: 除外 (loop=False)

■ 2種類のグラフ (split毎に構築)
  訓練グラフ: 1,523訓練患者間のみ
  検証グラフ: 1,904全患者
    ※ 検証患者は訓練患者の誤差を参照可 (データリーク防止)

■ 出力形式
  edge_index: [2, N×10]  → 隣接行列
  nn_idx:    [N, 10]    → 効率的な近傍ルックアップ
```

---

### スライド 11: 技術的課題① — DataFrameの排除

```
【Before / After 比較】

■ 問題 (Before)
  元のEPNモデルは pandas DataFrame を forward() に受け取り、
  列名から時間点を抽出していた
  → Lightning のテンソルベースパイプラインと非互換

■ 指導教員の指摘
  "This is Bad Practice"
  モデルはデータオブジェクトではなく
  次元とハイパーパラメータのみに依存すべき

【対比表】
  Before: forward(self, df: pd.DataFrame, nn_idx)
            ↑ DataFrameに依存 ✗

  After:  forward(self, features, surv_preds, labels,
                  nn_idx, test_idx, val_idx)
            ↑ テンソルのみ ✓

■ DataLoaderが返すバッチ形式
  {
    "features"   : Tensor [N, 9]    # 患者特徴量
    "surv_preds" : Tensor [N, T]    # MLP生存曲線
    "labels"     : Tensor [N, 2]    # (期間, イベント)
    "nn_idx"     : Tensor [N, 10]   # k-NN近傍インデックス
  }
```

---

### スライド 12: 技術的課題② — メモリ問題の解決

```
【問題の図解】

ナイーブなアプローチ: 全患者 × 全患者 × 時点数
  テンソル形状: [1523, 1523, 1391]
  必要メモリ: ~12.9 GB → OOMクラッシュ

【解決策: 2段階の最適化】

┌────────────────────────────────────────────────┐
│  最適化①: ミニバッチ処理                        │
│  256患者ずつランダムに選択して処理               │
│  [256, 256, 1391] → ~363 MB                    │
└────────────────────────────────────────────────┘
                    +
┌────────────────────────────────────────────────┐
│  最適化②: グラフによる注意の限定                │
│  全患者ではなくk=10近傍のみに注意を制限          │
│  [256, 10, 1391] → ~14 MB  ✓                  │
└────────────────────────────────────────────────┘

【メモリ比較表】
  方法                          テンソル形状        メモリ
  ─────────────────────────────────────────────────────
  全患者・完全注意 (ナイーブ)   [1523, 1523, 1391]  ~12.9 GB ✗
  ミニバッチ (256) + 完全注意   [256,  256,  1391]  ~363 MB
  ミニバッチ (256) + グラフ     [256,   10,  1391]  ~14 MB  ✓

  → グラフにより 26倍のメモリ削減を達成
```

---

### スライド 13: Hydra による設定管理

```
【設定ファイルの例】

# configs/experiment/run_epn.yaml
mlp:
  hidden_dim: 64
  epochs: 110
  optimizer:
    lr: 0.001
    weight_decay: 0.0005
epn:
  epochs: 50
  learning_rate: 0.001
  alpha: 0.0   # L1正則化
  beta:  0.0   # L2正則化
graph:
  k: 10
  metric: cosine

■ コマンドラインからパラメータを上書き可能
  # 近傍数を20に変更
  python -m ...run_epn_v2 graph.k=20

  # EPNの学習エポックを延長
  python -m ...run_epn_v2 epn.epochs=100

  → ソースコードを変更せずに再現可能な実験が可能
```

---

### スライド 14: 結果 — 患者類似度グラフの可視化

```
【図: patient_graph.png を挿入 (大きく配置)】

  ※ PCA 2次元投影
    - 青点: 打ち切り患者 (censored)
    - 橙点: イベント発生患者 (death)
    - 線: k=10近傍接続

■ 観察事項
  - 2グループは特徴空間全体に混在 → 明確な分離なし
    → C-indexが控えめな値になることと整合
  - 中心の密集クラスタ: 平均的な臨床プロファイルを持つ患者
  - 周辺の点: 極端な臨床特徴を持つ患者

PC1: 30.2% の分散を説明  /  PC2: 17.9% の分散を説明
1,523 訓練患者, k=10, コサイン類似度
```

---

### スライド 15: 結果 — 生存曲線の比較

```
【図: survival_curves.png を挿入 (大きく配置)】

4名の検証患者の生存曲線
  ─── MLP ベースライン (緑実線)
  - - EPN 補正後 (橙破線)
  ┊   観察されたイベント/打ち切り時刻 (点線縦軸)

■ 観察事項
  イベント患者 (上段 2名):
    EPN補正により、イベント時刻後に
    生存曲線が下方にシフトする傾向
    → 類似近傍の誤差シグナルを借用

  打ち切り患者 (下段 2名):
    補正はより穏やか
    → 打ち切り観察では誤差シグナルが弱いため
```

---

### スライド 16: 結果 — 定量的評価

```
【上: MLP vs EPN (split別)】

Split    MLP C-index   MLP IBS    EPN C-index   EPN IBS
──────────────────────────────────────────────────────
  1        0.6546       0.1718      0.5879        0.1818
  2        0.6486       0.1558      0.6345        0.1606
  3        0.6332       0.1777      0.6055        0.1822
  4        0.6578       0.1685      0.6347        0.1822
  5        0.6491       0.1605      0.6171        0.1778
──────────────────────────────────────────────────────
Mean±SD  0.6487±0.009  0.1669     0.6159±0.018  0.1769

【下: 3モデル比較】

  モデル                   Mean C-index     Mean IBS
  ─────────────────────────────────────────────────────
  SurvivalDGM             0.6043 ± 0.024   0.1754   ← 既存ベースライン
  MLP (Stage 1)           0.6495 ± 0.010   0.1664   ← 最良 ★
  EPN (このプロジェクト)   0.6157 ± 0.016   0.1769   ← SurvivalDGMを上回る

  EPN > SurvivalDGM ✓
  EPN < MLP  (さらなるチューニングが必要)
```

---

### スライド 17: 考察 — なぜEPNはMLPを上回らないか

```
■ MLPが強力なベースラインである理由
  - METABRIC は比較的小規模 (n≈1,900, 9特徴量)
  - 生存解析適切な損失関数 (CoxPH) + 適切な構造で十分
  - 大規模データセット (SUPPORT, n≈9,000) では
    グラフモデルの優位性が出やすい

■ EPNがMLPを上回らない要因
  ① ハイパーパラメータが未調整
     k=10, 50エポック, デフォルトlrのまま系統的な探索なし

  ② 特徴類似度 ≠ 誤差類似度
     9次元の臨床特徴が類似する患者が
     生物学的な複雑さにより同じ予測誤差を持つとは限らない

  ③ 学習データの少なさ
     ~1,523訓練患者では多様な誤差シグナルが不足

  ④ 損失関数の不一致
     MLP: CoxPH / EPN: CoxCC
     両者の貢献を独立して評価困難
```

---

### スライド 18: 今後の改善方向

```
改善領域               提案内容
──────────────────────────────────────────────────────────
グラフサイズ (k)       k ∈ {5, 10, 20, 50} のグリッドサーチ

学習エポック           100+エポックでのEPN収束確認

損失関数の統一         EPNにCoxPHを使用してMLPと直接比較

グラフ構築             特徴量重要度重み付き類似度
                       または学習済みグラフエッジ

データセット規模       SUPPORT (n≈9,000) で実験
                       → 人口ベース補正の効果が出やすい
```

---

### スライド 19: まとめ

```
■ 実装の成果 (エンジニアリング)
  ✓ EPNを完全にテンソルベースのLightningモジュールとして再実装
    (DataFrameフリー)
  ✓ メモリ使用量を ~12.9 GB → ~14 MB に削減 (26倍)
    (グラフ制約付き注意 + ミニバッチ)
  ✓ Hydraによる設定管理で再現可能な実験を実現
  ✓ モジュラーなコードベースでValentinのパイプラインに統合

■ 実験結果
  MLP           : C-index = 0.6495 (最良)
  EPN           : C-index = 0.6157 (SurvivalDGMを超える)
  SurvivalDGM   : C-index = 0.6043 (既存ベースライン)

■ 結論
  EPNはSurvivalDGMを上回ることを確認
  METABRICのような小規模データセットでは
  さらなるハイパーパラメータ調整が必要
  最も有望な改善: グラフ近傍サイズkの調整 + EPNエポック数の増加
```

---

### スライド 20: 参考文献

```
[1] Blondel, O. (2022). surv_epn: Error Propagation Network for survival analysis.
    INSA Lyon.

[2] Curtis, C. et al. (2012). The genomic and transcriptomic architecture
    of 2,000 breast tumours reveals novel subgroups. Nature, 486, 346-352.

[3] Kvamme, H., Borgan, O., & Scheel, I. (2019). Time-to-event prediction
    with neural networks and Cox regression. JMLR, 20(129), 1-30.

[4] Falcon, W. et al. (2019). PyTorch Lightning. GitHub.

[5] Fey, M. & Lenssen, J. E. (2019). Fast Graph Representation Learning
    with PyTorch Geometric. ICLR Workshop.

[6] Yao, D. et al. (2023). Hydra: A framework for elegant experiment
    management. Meta AI.

[7] Katzman, J. L. et al. (2018). DeepSurv: Cox proportional hazards
    deep neural network. BMC Medical Research Methodology, 18(1), 24.
```

---

## バックアップスライド (Q&A 対応用)

### B1: CoxPH損失の詳細

```
■ Cox比例ハザードモデル
  h(t|x) = h_0(t) × exp(f(x))    // f(x) = MLPが出力するリスクスコア

■ 部分尤度 (partial likelihood)
  L(β) = Π_i  exp(f(x_i)) / Σ_{j: t_j≥t_i} exp(f(x_j))

  → 打ち切りデータをリスクセットで正確に処理
  → イベント発生患者のみを分子に取る

■ CoxCC (Case-Control Cox)
  計算量削減のため完全なリスクセットの代わりに
  ランダムサンプリングしたコントロールを使用
```

### B2: k-NNグラフのデータリーク防止

```
■ 問題
  検証患者同士で誤差を参照し合うと
  検証セットの情報が漏れる (data leakage)

■ 解決策
  訓練グラフ: 訓練患者 → 訓練患者のみ
  検証グラフ: 検証患者 → 訓練患者のみ
    (検証患者は他の検証患者の誤差は参照不可)

  → val_nn_idx で参照先を訓練患者に限定
```

### B3: SurvivalDGM との比較詳細

```
■ SurvivalDGM の評価が特殊な理由
  グラフをエッジの有無で確率的にサンプリングするため
  推論のたびに異なるグラフ構造を使用

  → 安定したC-indexを得るために
    各splitで 100回のフォワードパスを実行し平均を取る

■ EPN との違い
  SurvivalDGM: グラフ構造自体を学習 (latent graph)
  EPN:         k-NNで構築した固定グラフ + 誤差補正
```

### B4: Hydra 設定ファイルの全体構成

```
configs/
├── experiment/
│   └── run_epn.yaml      # 実験全体の設定 (エントリポイント)
├── model/
│   ├── mlp.yaml          # MLP固有の設定
│   └── epn.yaml          # EPN固有の設定
├── data/
│   └── metabric.yaml     # データセット設定
├── trainer/
│   └── default.yaml      # Lightning Trainer 設定
└── callbacks/
    └── model_checkpoint.yaml  # チェックポイント保存
```

---

## 発表上の注意事項 (project_guidelines_ja.md より)

- **スライド番号を付ける** (質疑で「スライドN番について…」と言えるように)
- **20分を厳守** → リハーサルで時間を計測すること (時間超過で強制終了)
- **1枚のスライドに情報を詰め込みすぎない**
  - 特にスライド8 (リポジトリ構成) と12 (メモリ問題) は図を大きく見せる
- **図はできるだけ大きく配置** (patient_graph.png, survival_curves.png を活用)
- **Ubuntu上でPDFを確認** → Windows Coreフォント使用推奨 (Arial, Times New Roman等)
- **発表練習**: プロジェクトに関わっていない他の学生の前でリハーサルを行うこと
- **発表者間で時間を均等に配分** (Yosuke と Sena で約10分ずつ)
- **PDF形式で提出** (発表1日前までに hippocampus.ec-nantes.fr へ)
