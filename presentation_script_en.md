# Presentation Script — English
## Modernization and Integration of a New Graph-Based Deep Learning Method for Survival Analysis
**Yosuke Kimoto & Sena Fukabe — ECN 2026**

---

## SLIDE 01 — Title

Good morning, everyone.
My name is Yosuke Kimoto, and together with Sena Fukabe, we will present our project.
The topic is the integration of a graph-based deep learning method called the Error Propagation Network into an existing survival analysis pipeline.
We have about 20 minutes, so let's get started.

---

## SLIDE 02 — Agenda

Here is our plan for today.
We will start with background on survival analysis.
Then we explain our project objective, followed by an overview of the three models we worked with.
After that, we go into the implementation details.
Then we show our experimental results, and finish with discussion and future work.

---

## SLIDE 03 — Background: Survival Analysis

First, what is survival analysis?
The goal is to predict the survival function S(t|x) — the probability that a patient survives beyond a given time t.
The main challenge is censored data: patients who left the study before experiencing the event.
Standard machine learning cannot handle this directly.

We use two metrics.
The C-index measures ranking accuracy. 0.5 means random, and 1.0 is perfect.
The IBS measures the mean squared prediction error. Lower is better.

Our dataset is METABRIC — 1,904 breast cancer patients with 9 clinical features.
We use 5-fold cross-validation for all experiments.

---

## SLIDE 04 — Project Objective

Our main objective was to integrate EPN into Valentin's existing PyTorch Lightning pipeline — without simply copying the original code.

We followed five steps.
First, verify the existing SurvivalDGM pipeline.
Second, wrap the MLP as a Lightning module.
Third, build a patient similarity graph using PyTorch Geometric.
Fourth, reimplement EPN as a tensor-only Lightning module.
Fifth, add Hydra for configuration management.

One important feedback from our supervisor was:
passing a DataFrame to model.forward() is bad practice.
Models should depend only on tensors.
This shaped much of our implementation.

---

## SLIDE 05 — Three Models at a Glance

We worked with three models.

SurvivalDGM is the existing baseline from Valentin.
It uses a probabilistic graph and achieves a mean C-index of 0.6043.

Our MLP is Stage 1 of our pipeline.
It is a simple feedforward network: 9 → 64 → 64 → 1, trained with CoxPH loss.
It achieves the best result: 0.6495.

EPN is our main contribution — Stage 2 of the pipeline.
It is a graph attention network that corrects MLP predictions using neighbour errors.
It achieves 0.6157.

---

## SLIDE 06 — Stage 1: MLP Architecture

The MLP takes 9 clinical features as input.
It passes through two hidden layers of size 64 with ReLU activation.
The output is a risk score, which the PyCox library converts into survival curves.

The final output is a matrix of size N by T, where T is about 1,391 time points.
This matrix is the input to Stage 2.

We trained for 110 epochs with Adam, learning rate 0.001.

---

## SLIDE 07 — Stage 2: Error Propagation Network

The core idea of EPN is simple.
If the MLP makes similar mistakes for patients similar to patient A, we can use those mistakes to correct A's prediction.

The process has five steps.
First, find k=10 nearest neighbours in the patient graph.
Second, compute the MLP prediction error for each neighbour.
Third, compute attention weights using learned projections.
Fourth, compute a weighted sum of neighbour errors as the correction.
Fifth, add the correction to the MLP prediction.

The training loss is CoxCC — a case-control approximation of CoxPH.

---

## SLIDE 08 — Repository Structure

Here is the repository structure.
Gray files are Valentin's original code.
Blue files with a star are what we added.

We added graph_builder.py and epn_datamodule.py in the data layer.
We added mlp_module.py and epn_module.py as Lightning modules.
We added scripts for training, running the full pipeline, and visualization.

---

## SLIDE 09 — Two-Stage Pipeline

The pipeline runs in two stages.

In Stage 1, we train the MLP for 110 epochs.
After training, we call mlp.freeze() to lock the weights permanently.

In Stage 2, EPNDataModule precomputes MLP survival curves for all patients and builds the k-NN graphs.
Then we train the EPN for 50 epochs.

We repeat this for all 5 folds.

---

## SLIDE 10 — Patient Similarity Graph Construction

We use PyTorch Geometric to build the graph.
We compute cosine similarity between patients based on their 9 features.
Each patient gets k=10 nearest neighbours.

We build two graphs.
The training graph connects training patients only.
The validation graph connects validation patients to training patients only.
This prevents data leakage.

---

## SLIDE 11 — Engineering Challenge 1: DataFrame Removal

Our first challenge was the DataFrame dependency.
The original code passed a pandas DataFrame to forward().
This is incompatible with Lightning's tensor pipeline.

We refactored the interface to use only tensors:
features, survival predictions, labels, and neighbour indices.

---

## SLIDE 12 — Engineering Challenge 2: Memory Optimization

Our second challenge was memory.
The naïve approach requires about 12.9 gigabytes — this caused an out-of-memory crash.

We applied two optimizations.
First, mini-batching: process 256 patients per step instead of all 1,523.
Second, graph-constrained attention: attend only to k=10 neighbours, not all patients.

Together, these reduce memory to about 14 megabytes — a 26-times reduction.

---

## SLIDE 13 — Hydra Configuration Management

We used Hydra to manage hyperparameters through YAML files.
All settings for the MLP, EPN, and graph are in one config file.

The key benefit is that we can override any parameter from the command line.
No source code changes needed.
This makes experiments reproducible and easy to compare.

---

## SLIDE 14 — Patient Graph Visualization

This figure shows the patient similarity graph projected into 2D using PCA.
Blue points are censored patients, orange points experienced the event.

The two groups are interleaved — there is no clear separation in the feature space.
This explains the modest C-index values across all models.

---

## SLIDE 15 — Survival Curve Comparison

This figure shows survival curves for four patients.
Green is the MLP baseline, orange dashed is the EPN-corrected prediction.

For event patients in the top row, EPN shifts the curve downward after the event time.
The error signal from similar neighbours is working.

For censored patients in the bottom row, the correction is smaller.
Censored observations provide a weaker error signal.

---

## SLIDE 16 — Quantitative Results

Here are the numbers.

The MLP achieves a mean C-index of 0.6487, and EPN achieves 0.6159.

In the three-model comparison:
MLP is the best at 0.6495.
EPN achieves 0.6157 — better than SurvivalDGM's 0.6043.

So EPN surpasses the existing baseline, but does not yet beat the MLP.

---

## SLIDE 17 — Discussion

Why does EPN not surpass the MLP?

METABRIC is a relatively small dataset — about 1,900 patients with 9 features.
The MLP with CoxPH loss is already a strong baseline.
Graph-based methods tend to work better on larger datasets.

There are four specific reasons.
One: hyperparameters were not tuned.
Two: feature similarity does not always imply error similarity.
Three: 1,523 training patients may be too few for effective graph correction.
Four: MLP and EPN use different loss functions, making it hard to isolate the graph's contribution.

---

## SLIDE 18 — Future Improvements

We propose five directions.

Tune the graph size k over {5, 10, 20, 50}.
Train EPN for more epochs.
Use CoxPH for EPN for a fair comparison.
Explore learned graph construction.
Test on SUPPORT — about 9,000 patients — where graph methods should be more effective.

---

## SLIDE 19 — Conclusion

Our engineering achievements:
We reimplemented EPN as a tensor-only Lightning module.
We reduced memory from 12.9 GB to 14 MB — 26 times smaller.
We integrated Hydra for reproducible experiments.

Our results:
MLP: C-index = 0.6495 — best.
EPN: C-index = 0.6157 — beats the SurvivalDGM baseline.

EPN surpasses the baseline.
The most promising next step is tuning k and training duration.

Thank you. We are happy to take questions.

---

## SLIDE 20 — References

*(Display only — no script needed)*

---

## BACKUP SLIDES — Q&A

### B1 — CoxPH vs CoxCC

CoxPH uses all patients who have not yet failed as the risk set at each event time.
It handles censored data naturally, but is expensive for large datasets.

CoxCC replaces the full risk set with a small random sample of controls.
This reduces computation while approximating the CoxPH gradient.
It is the standard approach for large-scale survival deep learning.

### B2 — Data Leakage Prevention

If validation patients could attend to each other's errors, information from the validation set would leak into the model.

We prevent this with two separate graphs.
Training graph: training patients to training patients only.
Validation graph: validation patients to training patients only.
Validation patients cannot see each other.

### B3 — SurvivalDGM Evaluation

SurvivalDGM samples each edge from a Bernoulli distribution.
Every forward pass produces a different graph, so a single-pass C-index is unstable.
We average over 100 forward passes per fold.

EPN uses a fixed k-NN graph — deterministic, no repeated sampling needed.

### B4 — Hydra Config Tree

The config tree is composable.
You can mix and match config files and override any value from the command line.
No source code changes are needed for new experiments.
