# スライド作成プロンプト
## Modernization and Integration of a New Graph-Based Deep Learning Method for Survival Analysis

---

## 【このプロンプトの使い方】

このファイルはスライドを作成するためのマスタープロンプトです。
PowerPoint / Google Slides / LaTeX Beamer / Keynote のいずれかで実装してください。
各スライドの「コンテンツ」セクションをそのままスライドに転記し、
「デザイン仕様」セクションのスタイルを適用してください。

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 1: デザインシステム (全スライド共通)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 1-1. スライドサイズ

```
フォーマット: 16:9 ワイドスクリーン
サイズ: 33.87 cm × 19.05 cm  (PowerPoint 標準ワイド)
      または 1920 × 1080 px
```

### 1-2. カラーパレット

```
■ プライマリカラー (ECN ネイビー)
  HEX: #1B3A6B
  用途: ヘッダーバー背景、見出し文字、強調枠線

■ アクセントカラー (テクニカルブルー)
  HEX: #2E86C1
  用途: 図の強調、新規実装を示すマーカー、矢印

■ ハイライトカラー (アンバー)
  HEX: #E67E22
  用途: 重要数値の強調 (C-index 結果等)、★マーク

■ スライド背景 (オフホワイト)
  HEX: #F5F7FA
  用途: スライド全体の背景

■ コンテンツ背景 (純白)
  HEX: #FFFFFF
  用途: コードブロック、表のセル背景

■ 本文テキスト
  HEX: #1A1A2E
  用途: 本文・箇条書き

■ サブテキスト
  HEX: #5D6D7E
  用途: キャプション、補足説明

■ 成功カラー (グリーン)
  HEX: #27AE60
  用途: ✓ チェックマーク、良好な結果

■ 警告カラー (レッド)
  HEX: #E74C3C
  用途: ✗ エラー表示、OOM クラッシュ等
```

### 1-3. フォント設定

```
■ スライドタイトル (ヘッダーバー内)
  フォント: Arial Bold
  サイズ: 24pt
  色: #FFFFFF

■ スライド内の小見出し (セクション区切り)
  フォント: Arial Bold
  サイズ: 17pt
  色: #1B3A6B

■ 本文・箇条書き
  フォント: Arial Regular
  サイズ: 15pt
  色: #1A1A2E
  行間: 1.4

■ コード・技術テキスト
  フォント: Courier New Regular
  サイズ: 13pt
  色: #1A1A2E
  背景: #EAECEE (薄いグレー)

■ キャプション・補足
  フォント: Arial Italic
  サイズ: 11pt
  色: #5D6D7E

■ 強調数値 (C-index 等)
  フォント: Arial Bold
  サイズ: 18pt
  色: #E67E22

※ すべて Windows Core Fonts のみ使用 (Ubuntu 互換のため)
   使用可能フォント: Arial, Times New Roman, Courier New,
   Georgia, Trebuchet MS, Verdana
```

### 1-4. レイアウトグリッド

```
上部ヘッダーバー:
  高さ: 2.8 cm
  背景色: #1B3A6B
  内容: スライドタイトル (左揃え、左余白 1.0 cm)
        右端に発表タイトル略称 "EPN Integration"
        (Arial 11pt、白、透明度 60%)

コンテンツエリア:
  上辺: 3.2 cm から
  下辺: 17.8 cm まで
  左右余白: 各 1.2 cm

下部フッターバー:
  高さ: 1.2 cm
  上辺: 17.8 cm から
  背景色: #E8EDF4  (ECN ネイビーの 10% 薄め)
  左側: "Kimoto & Fukabe — ECN 2026"
         Arial 11pt、#5D6D7E
  中央: スライド番号 "01 / 20"
         Arial Bold 11pt、#1B3A6B
  右側: ECN ロゴ (詳細は 1-5 参照)
```

### 1-5. ECN ロゴの配置仕様

```
■ 配置位置: 各スライド右下フッターバー内
  右端から 0.5 cm 内側
  フッターバーの垂直中央

■ ロゴ仕様
  使用ファイル: Ecole Centrale de Nantes 公式ロゴ (PNG 透明背景)
  推奨: ネイビー版または白版を用途に応じて切り替え
  サイズ: 高さ 0.75 cm (フッター内に収まるよう調整)

■ ロゴ配色の使い分け
  タイトルスライド (スライド 1):
    白色ロゴ + "Ecole Centrale de Nantes" テキスト (白)
    → ネイビー背景上に配置
  それ以外のスライド:
    ネイビー (#1B3A6B) ロゴ
    → 薄いグレーのフッターバー (#E8EDF4) 上に配置

■ ロゴが用意できない場合の代替テキスト
  "ECN | Ecole Centrale de Nantes"
  Arial Bold 9pt、色: #1B3A6B
```

### 1-6. 繰り返し要素 (全スライド共通)

```
■ ヘッダーバー
  左端: 縦 4px 白線 + スライドタイトルテキスト (左余白 1.2 cm)
  右端: "EPN Integration" (Arial 11pt、白、透明度 60%)

■ フッターバー
  左: "Kimoto & Fukabe — ECN 2026"
  中: スライド番号
  右: ECN ロゴ

■ 記号の色規則 (全スライド統一)
  ▶ または ■: #2E86C1  (箇条書き先頭)
  ★: #E67E22  (新規実装ファイル、重要項目)
  ✓: #27AE60  (成功・達成)
  ✗: #E74C3C  (失敗・問題)
```

### 1-7. スライドテンプレート種別

```
Template A: タイトルスライド (スライド 1 のみ)
  全面グラデーション背景、図の装飾あり

Template B: 標準コンテンツ
  1 カラム、箇条書き・テキスト中心

Template C: 2 カラムレイアウト
  左: テキスト / 右: 図または表

Template D: 図中心
  大きな図 1 枚 + キャプション

Template E: 比較表
  Before/After または複数モデル比較

Template F: コードブロック付き
  技術的なスライド、等幅フォント領域を含む
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 2: 各スライドの詳細仕様
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

---

### SLIDE 01 — Title [Template A]

**レイアウト:**
```
背景: グラデーション (#1B3A6B → #2E5FA3、左から右)

左半分 (55%):
  上部: 白い横線 (2px) で装飾的なセパレータ
  メインタイトル: 白、Arial Bold 32pt、左揃え、上余白 3.5cm
  著者名: 白、Arial Bold 18pt、メインタイトルの下
  所属: 白、Arial Regular 14pt
  日付: 白、Arial Regular 14pt

右半分 (45%):
  patient_graph.png を透明度 35% で背景的に配置
  (装飾目的のみ。視認性は不要)

右下:
  ECN ロゴ (白) + "Ecole Centrale de Nantes" (白 Arial Bold 11pt)
  フッターバーではなくスライド内に大きめに配置
```

**コンテンツ:**
```
Modernization and Integration of a New
Graph-Based Deep Learning Method
for Survival Analysis

Yosuke Kimoto    Sena Fukabe

ECN — Ecole Centrale de Nantes
June 2026
```

---

### SLIDE 02 — Agenda [Template B]

**レイアウト:**
```
ヘッダーバー: "Agenda"

コンテンツ: 番号付きリスト
  番号: 色付き円バッジ (直径 0.8cm、背景 #2E86C1、白数字 Bold 14pt)
  テキスト: Arial Regular 17pt、#1A1A2E
  項目間余白: 0.4cm
  サブ項目: 左インデント 1.2cm、Arial 14pt、#5D6D7E
```

**コンテンツ:**
```
1   Background: Survival Analysis

2   Project Objective

3   Models Overview  (MLP / EPN / SurvivalDGM)

4   Implementation
      • Repository Structure
      • Two-Stage Pipeline
      • Engineering Challenges & Solutions

5   Experimental Results

6   Discussion & Future Work
```

---

### SLIDE 03 — Background: Survival Analysis [Template C]

**レイアウト:**
```
ヘッダーバー: "Background: Survival Analysis"

左カラム (55%):
  3 つのセクション (セクションの前に #1B3A6B 縦 2px 細線)
  セクション①: 目的と打ち切りの説明
  セクション②: 評価指標 (C-index / IBS)

右カラム (40%):
  上: 生存曲線の概念図
    2 本の曲線 (High risk / Low risk)
    X 軸: Time、Y 軸: Survival probability S(t)
    打ち切り患者の "+" マーク
  下: METABRIC 情報ボックス
    角丸四角形、背景 #EBF5FB、枠線 #2E86C1
```

**コンテンツ:**
```
[左カラム]
■ What is Survival Analysis?
  • Predict S(t|x): probability that a patient
    survives beyond time t
  • Challenge: censored observations
    — patients who left the study without the event
  • Standard ML cannot handle censoring directly

■ Evaluation Metrics
  C-index (Concordance Index)
    Ranking accuracy of predicted risk
    0.5 = random  /  1.0 = perfect

  IBS (Integrated Brier Score)
    Mean squared error of predictions vs. outcomes
    Lower is better

[右下ボックス]
Dataset: METABRIC
─────────────────────────
Patients:     1,904
Features:     9 clinical variables (x0–x8)
Event rate:   ~58%  (death = 1)
Evaluation:   5-fold cross-validation
Preprocessing: StandardScaler (continuous features)
```

---

### SLIDE 04 — Project Objective [Template B]

**レイアウト:**
```
ヘッダーバー: "Project Objective"

上部 (35%): 主目的 — 強調ボックス
  角丸四角形、背景 #EBF5FB、左辺太線 4px #1B3A6B

下部 (60%): 段階的ステップ — フロー図
  各ステップ: 丸バッジ (番号) + テキスト、矢印でつなぐ
  縦フロー配置
```

**コンテンツ:**
```
[主目的ボックス]
Integrate the Error Propagation Network (EPN)
into Valentin's existing PyTorch Lightning pipeline
— without copy-pasting the original code

[ステップフロー]
Step 1  Verify existing SurvivalDGM pipeline (5-fold CV)
   ↓
Step 2  Wrap Oriane's MLP as a Lightning module
   ↓
Step 3  Build patient similarity graph (PyTorch Geometric)
   ↓
Step 4  Reimplement EPN as a tensor-only Lightning module
   ↓
Step 5  Integrate Hydra configuration management

[下部注記: 黄色ハイライトボックス]
★ Supervisor feedback:
  "DataFrames in model.forward() = Bad Practice"
  Models must depend only on tensor dimensions, not data objects.
```

**デザインメモ:**
```
ステップ番号バッジ: 背景 #1B3A6B、白数字、直径 0.85cm
矢印: #2E86C1、幅 2px
注記ボックス: 背景 #FEF9E7、枠線 #E67E22 左辺 4px
```

---

### SLIDE 05 — Three Models at a Glance [Template E]

**レイアウト:**
```
ヘッダーバー: "Three Models at a Glance"

コンテンツ: 3 カラム比較
  カラム間: 縦線 1px #BDC3C7

各カラム上部: モデル名バッジ (横幅いっぱい)
  SurvivalDGM → 背景 #7F8C8D (グレー: 既存)
  MLP         → 背景 #1B3A6B (ネイビー: Stage 1)
  EPN         → 背景 #2E86C1 (ブルー: このプロジェクト)

各カラム本文: 種別・特徴・C-index
C-index 数値: Arial Bold 20pt、#E67E22
```

**コンテンツ:**
```
[カラム 1: SurvivalDGM]
Probabilistic Graph Model
─────────────────────────
Existing baseline (Valentin)
Stochastic latent graph
Requires 100 forward passes
per split for stable C-index
─────────────────────────
Mean C-index:  0.6043

[カラム 2: MLP]  ← ★ BEST バッジ付き
Feedforward Neural Network
─────────────────────────
Stage 1 of our pipeline
Architecture:
  Input(9) → 64 → 64 → 1
Loss: CoxPH
─────────────────────────
Mean C-index:  0.6495

[カラム 3: EPN]  ← "(This project)" ラベル付き
Graph Attention Network
─────────────────────────
Stage 2 of our pipeline
Corrects MLP predictions
using k-NN neighbour errors
Loss: CoxCC
─────────────────────────
Mean C-index:  0.6157
```

---

### SLIDE 06 — Stage 1: MLP Architecture [Template C]

**レイアウト:**
```
ヘッダーバー: "Stage 1: MLP Architecture"

左カラム (45%): ネットワーク縦フロー図
  各層を角丸四角形で表現
  矢印: #2E86C1

右カラム (50%):
  上: 学習設定テーブル (2列)
  下: 損失関数の簡単な説明ボックス
```

**コンテンツ:**
```
[左: ネットワーク図]
┌──────────────────┐
│  Input  (dim=9)  │  9 clinical features
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Linear(9 → 64)   │
│    + ReLU        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Linear(64 → 64)  │
│    + ReLU        │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  Linear(64 → 1)  │  → risk score
└────────┬─────────┘
         ↓
   PyCox CoxPH wrapper
         ↓
   S(t|x)  [N × T]    full survival curves

[右上: 設定テーブル]
Parameter       Value
─────────────────────
Loss function   CoxPH
Epochs          110
Optimizer       Adam
Learning rate   0.001
Weight decay    0.0005
Evaluation      5-fold CV

[右下: 補足]
Output S(t|x): survival probability
at ~1,391 discrete time points
per patient  → used as EPN input
```

---

### SLIDE 07 — Stage 2: Error Propagation Network [Template B]

**レイアウト:**
```
ヘッダーバー: "Stage 2: Error Propagation Network (EPN)"

上部 (25%): コアアイデアボックス
  背景 #FEF9E7、枠線左辺 #E67E22 4px

下部 (70%): 5 ステップ補正フロー
  各ステップ: 番号バッジ + テキスト + 数式 (Courier New 13pt)
  ステップ間: 点線区切り #BDC3C7
```

**コンテンツ:**
```
[コアアイデアボックス]
Core Idea:
If the MLP consistently misestimates survival for patients similar
to patient A, we can correct A's prediction using those errors.

[5 ステップ補正フロー]
① Find k=10 nearest neighbours in the k-NN graph

② Compute MLP prediction error for each neighbour j:
     error_j(t) = true_survival_j(t) − MLP_pred_j(t)

③ Compute attention weights (learned Key / Query projections):
     att_j = softmax( (W_q · feat_A) · (W_k · feat_j) / √F )

④ Compute correction term:
     correction(t) = Σ_j  att_j × error_j(t)

⑤ Corrected prediction:
     S_corrected_A(t) = MLP_pred_A(t) + correction(t)

Training loss: CoxCC (Case-Control Cox)
```

---

### SLIDE 08 — Repository Structure [Template F]

**レイアウト:**
```
ヘッダーバー: "Repository Structure"

全幅の大きなコードブロック風エリア (白背景、グレー枠線 1px)
Courier New 13pt、行間 1.3

色分けルール:
  既存ファイル: #7F8C8D (グレー)
  新規ファイル: #2E86C1 (ブルー) + "★ NEW" バッジ (#E67E22 背景、白字 9pt)
  フォルダカテゴリラベル: #1B3A6B Bold

下部: 凡例バー (2 項目)
  ■ グレー = Existing (Valentin's code)
  ★ ブルー = Added by this project
```

**コンテンツ:**
```
Vanlentin's pipeline/
│
├── configs/                              [Configuration]
│   ├── experiment/
│   │   └── ★ run_epn.yaml              Hydra main config
│   └── model/ { mlp.yaml, epn.yaml }
│
└── src/survival_analysis/
    │
    ├── data/                             [Data Layer]
    │   ├──   metabric_datamodule.py     Load & preprocess METABRIC
    │   ├── ★ graph_builder.py           Build k-NN patient similarity graph
    │   └── ★ epn_datamodule.py         Pre-compute MLP curves + graph
    │
    ├── models/                           [Model Layer]
    │   ├── ★ mlp_module.py             MLP as LightningModule
    │   ├── ★ epn_module.py             EPN with graph attention + CoxCC
    │   └──   dgm_model.py              SurvivalDGM (existing)
    │
    └── experiments/                      [Experiment Scripts]
        ├── ★ run_mlp.py                Train MLP (5 splits)
        ├── ★ run_epn_v2.py             Full MLP+EPN pipeline (Hydra entry)
        └── ★ visualize.py              Graph & survival curve plots

─────────────────────────────────────────────────────────────
  (gray) = Valentin's original code       ★ (blue) = Added by this project
```

---

### SLIDE 09 — Two-Stage Pipeline [Template B]

**レイアウト:**
```
ヘッダーバー: "Two-Stage Pipeline Architecture"

上半分: Stage 1 ブロック
  背景: #EBF5FB、左辺太線 4px #1B3A6B
  タイトル: "Stage 1: MLP Training"、Arial Bold 15pt、#1B3A6B

中央の矢印エリア:
  ↓ + "mlp.freeze() — weights permanently locked"
  テキスト: Arial Bold 13pt、#E67E22

下半分: Stage 2 ブロック
  背景: #E8F8F5、左辺太線 4px #2E86C1
  タイトル: "Stage 2: EPN Training"、Arial Bold 15pt、#2E86C1

右端: 小さなデータ分割ボックス
```

**コンテンツ:**
```
[Stage 1 ブロック]
  DataModule:  MetabricGraphSurvivalDataModule  (METABRIC, split i)
    ↓
  MLPModule:   110 epochs  |  CoxPH loss  |  Adam lr=0.001
    ↓
  Evaluate:    CoxPH wrapper → C-index, IBS

       ↓  mlp.freeze()  — weights permanently locked

[Stage 2 ブロック]
  EPNDataModule.setup():
    • Pre-compute MLP survival curves for all patients   [N_all, T]
    • Build training k-NN graph   → train_nn_idx  [1523, 10]
    • Build validation k-NN graph → val_nn_idx    [1904, 10]
    ↓
  EPNModule:   50 epochs  |  CoxCC loss  |  Adam lr=0.001
    ↓
  Evaluate:    EvalSurv → C-index, IBS

[右: データ分割ボックス]
5-fold CV
───────────
Train: ~1,523
Val:    ~381
```

---

### SLIDE 10 — Patient Similarity Graph Construction [Template C]

**レイアウト:**
```
ヘッダーバー: "Patient Similarity Graph Construction"

左カラム (40%):
  グラフの概念図 (患者ノード + エッジ)
  ノード: 丸、イベント患者 = #E67E22、打ち切り = #2E86C1
  エッジ: 細い灰色線

右カラム (55%):
  上: 実装パラメータ (2列テーブル)
  下: 2種類のグラフの違い (簡単なフロー説明)
```

**コンテンツ:**
```
[左: 概念図]
       ● feat_B
      /
 ●──── Patient A ────●  feat_D
      \          (k=10 neighbours)
       ● feat_C

Metric: cosine similarity
(scale-invariant across features)

[右上: パラメータテーブル]
Parameter       Value
──────────────────────
Library         PyTorch Geometric (PyG)
Metric          Cosine similarity
Neighbours (k)  10 per patient
Self-loop       Excluded

Output:
  edge_index [2, N×10]  adjacency
  nn_idx     [N, 10]    neighbour lookup

[右下: 2種類のグラフ]
Training graph:
  1,523 train patients → train patients only

Validation graph:
  1,904 all patients
  Val patients attend to TRAIN errors only
  ★ Prevents data leakage
```

---

### SLIDE 11 — Engineering Challenge 1: DataFrame Removal [Template E]

**レイアウト:**
```
ヘッダーバー: "Engineering Challenge ①: Removing the DataFrame Dependency"

上部 (20%): 問題提起ボックス
  背景: #FDEDEC、枠線左辺: #E74C3C 4px

中部 (55%): Before / After 2カラム比較
  Before カラム: 背景 #FDEDEC
  After カラム:  背景 #EAFAF1
  各カラム上部にバッジ: "✗ Bad Practice" / "✓ Refactored"

下部 (20%): DataLoader バッチ形式
  コードブロック風、Courier New 12pt
```

**コンテンツ:**
```
[問題提起ボックス]
⚠  Supervisor feedback:
"Passing a DataFrame to model.forward() is Bad Practice.
 Models must depend only on tensor dimensions and hyperparameters."

[Before カラム]  ✗ Bad Practice
forward(self,
  df: pd.DataFrame,   ← DataFrame!
  nn_idx
)
# Parses column names
# to extract time points
# → incompatible with
#   Lightning's tensor pipeline

[After カラム]  ✓ Refactored
forward(self,
  features,    # [N, 9]
  surv_preds,  # [N, T]
  labels,      # [N, 2]
  nn_idx,      # [N, 10]
  test_idx,
  val_idx
)
# Pure tensor interface ✓

[下部: DataLoader バッチ形式]
DataLoader yields:
{ "features"   : Tensor [N,  9],   # clinical features
  "surv_preds" : Tensor [N,  T],   # MLP survival curves
  "labels"     : Tensor [N,  2],   # (duration, event)
  "nn_idx"     : Tensor [N, 10] }  # k-NN indices
```

---

### SLIDE 12 — Engineering Challenge 2: Memory Optimization [Template B]

**レイアウト:**
```
ヘッダーバー: "Engineering Challenge ②: Out-of-Memory Problem"

上部 (20%): 問題の提示
  赤いボックス (背景 #FDEDEC)

中部 (40%): 2 つの解決ボックスを "+" でつなぐ
  Box1: 背景 #EBF5FB
  Box2: 背景 #E8F8F5

下部 (35%): 比較テーブル
  3行のテーブル、最終行 (正解) をハイライト
```

**コンテンツ:**
```
[問題ボックス]
Naïve approach: compute attention over ALL patients simultaneously
  Tensor: [1523, 1523, 1391]  →  ~12.9 GB  ✗  OOM CRASH

[解決策 Box1]
Optimization ①: Mini-batching
  Process 256 randomly selected training
  patients per step (not all 1,523)
  → [256, 256, 1391]  ≈  363 MB

                    +

[解決策 Box2]
Optimization ②: Graph-constrained attention
  Attend only to k=10 pre-selected
  neighbours (not all N patients)
  → [256,  10, 1391]  ≈  14 MB  ✓

[比較テーブル]
Method                        Tensor Shape           Memory
───────────────────────────────────────────────────────────
All patients, full attention   [1523, 1523, 1391]    ~12.9 GB  ✗
Mini-batch (256), full att.    [ 256,  256, 1391]    ~363 MB
Mini-batch (256) + graph  ★    [ 256,   10, 1391]    ~14 MB   ✓
───────────────────────────────────────────────────────────
Result: 26× memory reduction vs. mini-batch alone
```

**デザインメモ:**
```
"12.9 GB / OOM CRASH": Arial Bold 15pt、#E74C3C
"14 MB ✓": Arial Bold 15pt、#27AE60
"26×": Arial Bold 22pt、#E67E22 (最も目立つ要素)
テーブルヘッダー行: 背景 #1B3A6B、白テキスト
最終行: 背景 #EAFAF1
```

---

### SLIDE 13 — Hydra Configuration Management [Template F]

**レイアウト:**
```
ヘッダーバー: "Hydra Configuration Management"

左カラム (55%): YAML コードブロック (Courier New 12pt、背景 #EAECEE)
右カラム (40%): CLI 使用例 + メリット説明
```

**コンテンツ:**
```
[左: YAML コードブロック]
# configs/experiment/run_epn.yaml

mlp:
  hidden_dim: 64
  epochs:     110
  optimizer:
    lr:           0.001
    weight_decay: 0.0005

epn:
  epochs:        50
  learning_rate: 0.001
  alpha: 0.0    # L1 regularization
  beta:  0.0    # L2 regularization

graph:
  k:      10
  metric: cosine

[右: CLI 例]
Override from command line
— no source code changes:

# Change neighbourhood size
python -m ...run_epn_v2 \
    graph.k=20

# Extend EPN training
python -m ...run_epn_v2 \
    epn.epochs=100

✓ Reproducible experiments
✓ Consistent with Valentin's
  existing pipeline
```

---

### SLIDE 14 — Results: Patient Graph Visualization [Template D]

**レイアウト:**
```
ヘッダーバー: "Results: Patient Similarity Graph Visualization"

メイン: patient_graph.png を全幅 70% の高さで大きく配置
  枠線: 薄いグレー 1px

下部または右サイド: キャプション + 観察事項
```

**コンテンツ:**
```
[メイン図]
patient_graph.png
(挿入: Vanlentin's pipeline/patient_graph.png)

[キャプション]
Figure 1: Patient Similarity Graph (k-NN, k=10, cosine similarity)
PCA 2D projection — PC1: 30.2%, PC2: 17.9%
● Censored patients (blue)   ● Event patients / death (orange)

[観察事項]
• Event and censored groups are interleaved
  → No clear separation in clinical feature space
  → Consistent with modest C-index values across all models

• Dense cluster (centre): patients with average clinical profiles
• Peripheral points: patients with extreme clinical features
```

---

### SLIDE 15 — Results: Survival Curve Comparison [Template D]

**レイアウト:**
```
ヘッダーバー: "Results: Survival Curve Comparison"

メイン: survival_curves.png を大きく配置 (幅の 80% 程度)

下部: 凡例 + 観察事項 (2段)
```

**コンテンツ:**
```
[メイン図]
survival_curves.png
(挿入: Vanlentin's pipeline/survival_curves.png)

[凡例]
── MLP baseline (green solid line)
- - EPN corrected prediction (orange dashed line)
┊   Observed event / censoring time (dotted vertical line)

[観察事項]
Event patients (top row, 2 patients):
  EPN shifts the survival curve downward after the event time
  → Error signal borrowed from similar neighbours

Censored patients (bottom row, 2 patients):
  Correction is more moderate
  → Weaker error signal for censored observations
```

---

### SLIDE 16 — Results: Quantitative Performance [Template E]

**レイアウト:**
```
ヘッダーバー: "Results: Quantitative Performance"

上部 (45%): MLP vs EPN の split 別テーブル
  6行 (split 1-5 + Mean±SD 行)
  Mean 行: 背景 #EBF5FB、Bold

下部 (50%): 3 モデル比較テーブル
  4行、MLP 行にハイライト
  下部に結論の 2 行サマリー
```

**コンテンツ:**
```
[上: MLP vs EPN (per split)]
Split   MLP C-index   MLP IBS    EPN C-index   EPN IBS
──────────────────────────────────────────────────────
  1       0.6546       0.1718      0.5879        0.1818
  2       0.6486       0.1558      0.6345        0.1606
  3       0.6332       0.1777      0.6055        0.1822
  4       0.6578       0.1685      0.6347        0.1822
  5       0.6491       0.1605      0.6171        0.1778
──────────────────────────────────────────────────────
Mean    0.6487±0.009   0.1669    0.6159±0.018   0.1769

[下: Three-Model Comparison]
Model                  Mean C-index     Mean IBS
────────────────────────────────────────────────────
SurvivalDGM            0.6043±0.024    0.1754   ← existing baseline
MLP (Stage 1)  ★       0.6495±0.010    0.1664   ← best
EPN (this work)        0.6157±0.016    0.1769   ← > SurvivalDGM ✓
────────────────────────────────────────────────────
EPN > SurvivalDGM  ✓         EPN < MLP  (further tuning needed)
```

---

### SLIDE 17 — Discussion [Template B]

**レイアウト:**
```
ヘッダーバー: "Discussion: Why EPN Does Not Yet Surpass the MLP"

上部 (30%): MLP の強さの説明ボックス
  背景 #EBF5FB、枠線左辺 #1B3A6B

下部 (65%): 4 つの要因 (番号付きリスト)
  各要因の先頭: 番号バッジ (#2E86C1) + 太字の短い見出し + 本文
```

**コンテンツ:**
```
[MLP の強さ]
The MLP is a strong baseline for METABRIC (n≈1,900, 9 features).
Small dataset + appropriate survival loss (CoxPH) = already competitive.
Graph-based population methods tend to excel on larger datasets
(e.g., SUPPORT: n≈9,000).

[4 つの要因]
① Hyperparameters are untuned
   k=10, 50 epochs, default lr — no systematic search performed

② Feature similarity ≠ error similarity
   Clinically similar patients (9D feature space) may differ
   in prediction errors due to complex biological interactions

③ Limited training data
   ~1,523 training patients may not provide
   diverse enough error signals for graph correction

④ Loss function mismatch
   MLP uses CoxPH; EPN uses CoxCC
   → Difficult to isolate the graph correction's contribution
```

---

### SLIDE 18 — Future Improvements [Template B]

**レイアウト:**
```
ヘッダーバー: "Future Improvements"

コンテンツ: 5 行の表形式
  左列: 改善領域 (Arial Bold 14pt、#1B3A6B)
  右列: 具体的な提案 (Arial Regular 14pt)
  行間: グレー細線で区切り
```

**コンテンツ:**
```
Area                    Proposed Improvement
───────────────────────────────────────────────────────────
Graph size (k)          Grid search: k ∈ {5, 10, 20, 50}
                        Larger k may provide richer error signals

Training epochs         EPN convergence test with 100+ epochs

Loss alignment          Use CoxPH for EPN to enable
                        direct comparison with MLP baseline

Graph construction      Feature-importance-weighted similarity
                        or end-to-end learned graph edges

Dataset scale           Test on SUPPORT (n≈9,000)
                        where population-based corrections
                        are more effective
```

---

### SLIDE 19 — Conclusion [Template C]

**レイアウト:**
```
ヘッダーバー: "Conclusion"

左カラム (52%): 実装の成果 (緑チェックリスト)
右カラム (43%): 実験結果サマリー + 結論ボックス
  結論ボックス: 角丸四角形、背景 #EBF5FB、枠線左辺 #1B3A6B 4px
```

**コンテンツ:**
```
[左: Engineering Achievements]
✓ EPN fully reimplemented as tensor-only
  LightningModule
  (DataFrame-free, no surv_epn runtime imports)

✓ Memory: ~12.9 GB → ~14 MB  (26× reduction)
  via graph-constrained attention + mini-batching

✓ Hydra configuration management integrated
  → reproducible, command-line-configurable experiments

✓ Modular codebase consistent with
  Valentin's existing pipeline conventions

[右上: Results]
MLP (Stage 1)   C-index = 0.6495  ★ Best
EPN (this work) C-index = 0.6157
SurvivalDGM     C-index = 0.6043  (baseline)

[右下: 結論ボックス]
EPN surpasses SurvivalDGM baseline.
For small datasets like METABRIC,
hyperparameter tuning is essential.
Most promising next step:
tuning graph size k and EPN epochs.
```

---

### SLIDE 20 — References [Template B]

**レイアウト:**
```
ヘッダーバー: "References"

コンテンツ: 番号付きリスト
  フォントサイズ: 12pt (通常より小さく)
  行間: 1.7
  番号色: #2E86C1
```

**コンテンツ:**
```
[1] Blondel, O. (2022). surv_epn: Error Propagation Network for survival analysis.
    INSA Lyon.

[2] Curtis, C. et al. (2012). The genomic and transcriptomic architecture
    of 2,000 breast tumours reveals novel subgroups. Nature, 486, 346–352.

[3] Kvamme, H., Borgan, O., & Scheel, I. (2019). Time-to-event prediction
    with neural networks and Cox regression. JMLR, 20(129), 1–30.

[4] Falcon, W. et al. (2019). PyTorch Lightning. GitHub.

[5] Fey, M. & Lenssen, J. E. (2019). Fast Graph Representation Learning
    with PyTorch Geometric. ICLR Workshop.

[6] Yao, D. et al. (2023). Hydra: A framework for elegant experiment
    management. Meta AI.

[7] Katzman, J. L. et al. (2018). DeepSurv: personalized treatment
    recommender system using Cox deep neural network.
    BMC Medical Research Methodology, 18(1), 24.
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 3: バックアップスライド (Q&A 対応)
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

バックアップスライドはメイン 20 枚の後に配置。
スライド番号は "B1", "B2" 等と表記 (フッターバー左側)。
ヘッダーバー背景色を #7F8C8D に変更してバックアップスライドと明示。

---

### SLIDE B1 — CoxPH vs CoxCC Details [Template F]

**コンテンツ:**
```
[CoxPH — used by MLP]
Partial likelihood:
  L(β) = Π_i  exp(f(x_i)) / Σ_{j: t_j ≥ t_i} exp(f(x_j))

Risk set = all patients not yet failed at time t_i
→ Handles censored data naturally
→ Computationally expensive for large N

[CoxCC — used by EPN]
Case-Control approximation of CoxPH:
• Replace full risk set with random sample of "controls"
• Reduces computation while approximating CoxPH gradient
• Standard approach for large-scale survival deep learning
```

---

### SLIDE B2 — Data Leakage Prevention [Template B]

**コンテンツ:**
```
Problem:
  If validation patients attend to each other's errors
  → information from the val set leaks into the model

Solution:
  Training graph:
    Train patients → Train patients only
    (intra-training neighbourhood)

  Validation graph:
    Val patients → Train patients only
    ★ Val patients CANNOT attend to other val patients

Implementation in EPNDataModule.setup():
  val_nn_idx indices always point to training patients
  → no information leakage across the train/val boundary
```

---

### SLIDE B3 — SurvivalDGM Evaluation Details [Template B]

**コンテンツ:**
```
Why SurvivalDGM requires 100 forward passes per split:

SurvivalDGM learns a latent graph:
  Each edge is sampled stochastically (Bernoulli)
  → Every forward pass produces a different graph structure
  → Single-pass C-index is unstable

Solution: average C-index over 100 forward passes per split

EPN uses a fixed k-NN graph:
  → Deterministic: same graph every pass
  → Single forward pass = stable C-index
  → No repeated sampling required
```

---

### SLIDE B4 — Full Hydra Config Tree [Template F]

**コンテンツ:**
```
configs/
├── experiment/
│   └── run_epn.yaml           # Main experiment config (Hydra entry)
├── model/
│   ├── mlp.yaml               # MLP-specific hyperparameters
│   └── epn.yaml               # EPN-specific hyperparameters
├── data/
│   └── metabric.yaml          # Dataset & cross-validation settings
├── trainer/
│   ├── default.yaml           # Lightning Trainer defaults
│   └── gpu.yaml               # GPU-specific override
└── callbacks/
    ├── model_checkpoint.yaml  # Checkpoint saving config
    └── early_stopping.yaml    # Early stopping config

Composability:
  python -m ...run_epn_v2 trainer=gpu data=metabric graph.k=20
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 4: 最終チェックリスト
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### デザイン確認
- [ ] 全スライドのヘッダーバーが #1B3A6B で統一されている
- [ ] フォントがすべて Arial / Courier New のみ (Ubuntu 互換)
- [ ] 各スライド右下フッターバーに ECN ロゴが配置されている
- [ ] フッターバー中央にスライド番号が表示されている
- [ ] 1枚のスライドに情報を詰め込みすぎていない (箇条書き 6 行以内を目安)
- [ ] 図 (patient_graph.png, survival_curves.png) が挿入されている

### コンテンツ確認
- [ ] 発表は英語で統一されている
- [ ] C-index / IBS の数値が正確である (スライド 16 参照)
- [ ] コードブロックは Courier New で区別されている
- [ ] スライド 8 のリポジトリ図の色分けが正確 (既存 = グレー、新規 = ブルー)

### 発表要件 (project_guidelines_ja.md より)
- [ ] PDF 形式でエクスポート済み
- [ ] Ubuntu / Linux 上で文字化け・レイアウト崩れがないか確認
- [ ] 20 スライドで約 20 分に収まること (リハーサルで計測)
- [ ] バックアップスライドはメイン 20 枚の後に配置

### 提出要件
- [ ] ファイル名: `Presentation_Kimoto_Fukabe_2026.pdf`
- [ ] 発表 1 日前までに hippocampus.ec-nantes.fr にアップロード
- [ ] 暫定版を早めにアップロードし、後から更新可能
