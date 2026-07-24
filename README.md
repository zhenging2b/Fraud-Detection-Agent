# Fraud Detection Agent

Transaction fraud classification on the [IEEE-CIS Kaggle dataset](https://www.kaggle.com/c/ieee-fraud-detection) (~590k transactions, 3.5% fraud) paired with an LLM agent pipeline that explains *why* a flagged transaction was predicted fraudulent — mirroring how a fraud-ops analyst would justify escalating a case.

**Status: proof of concept.** The current, actively-developed pipeline lives entirely in
one notebook, `src/features/feature_engineeringupdated.ipynb`; most `.py` files under
`src/model/` predate it and aren't imported by the current path (see Project layout below).

## Architecture

```
Transaction
    │
    ▼
[Risk Model]           LightGBM + XGBoost + CatBoost, trained on a temporal 80/20 split
    │                  → fraud_probability
    ▼
[SHAP explanation]     SHAPExplainer (business-labeled, category-decoded)
    │
    ▼
[Main agent]           gathers findings from two subagents-as-tools:
  • SHAP subagent         explains which features drove the prediction
  • Similarity subagent   finds confirmed-fraud cases resembling this transaction
    │
    ▼
[Judge]                reviews both subagents' findings for agreement/unsupported
                        claims, produces the final justification (no tools of its own)
    │
    ▼
Justification text: verdict (strong/moderate/weak) + supporting points + caveats
```

The main agent does **not** synthesize a final answer itself — it only gathers both
subagents' raw output. A separate judge reviews that output and produces what actually
reaches the analyst. This is a deliberate two-stage design, not the triage→investigation→report
state machine originally planned (see CLAUDE.md's Architecture section for the full detail
and file-level pointers).

### Design choices worth knowing

**Why LightGBM/XGBoost/CatBoost, not one model?**
Ranking quality on held-out data is close between the three; ensembling by rank-averaging
their scores measurably beats any single model on the true-prevalence holdout (ensemble
PR-AUC 0.5213 vs. LightGBM alone 0.5194). Kept as a rank-averaged ensemble, not a trained
meta-learner (the meta-learner path exists in `src/model/train.py` but isn't run from the
current notebook).

**Why AUPRC, not AUROC?**
At 3.5% fraud prevalence, a trivial classifier achieves AUROC > 0.90 while catching almost
no fraud. AUPRC directly measures precision-recall quality at the operating point.
**AUPRC is also not comparable across different fraud-prevalence rates** — a public Kaggle
notebook's ~0.72 PR-AUC (kept in `src/features/fraud-detection.ipynb` for reference)
comes from evaluating on an artificially upsampled ~12%-fraud subsample, not a better
model; the number to actually compare against is this repo's own 0.52 at the true ~3.5%
rate.

**Why temporal CV, not random k-fold?**
The competition's real test set is 90 days after training. Random k-fold leaks future
data into past folds and inflates AUPRC. The notebook sorts by `TransactionDT` and takes
a fixed 80/20 cutoff — train 472,432 rows, test 118,108 rows (test fraud rate 3.44%).
Every train-only-fit step (frequency encoding, velocity aggregates) had to be explicitly
refit after the split at least twice during development — computing them on the full
pre-split dataframe is a real, easy-to-reintroduce leak.

**Why not one-hot encode categoricals?**
All three tree models handle integer-coded categoricals natively. One-hot would explode
dimensionality (`DeviceInfo` alone has ~1,787 unique values) and fragment SHAP
explanations across many binary columns instead of one clean value. The code→category
mapping is preserved separately (`category_maps`) so SHAP output can still show the
human-readable value (`card6 = "credit"`), without the model itself needing one-hot input.

## Setup

```bash
conda env create -f environment.yml
conda activate fraud_d
```

**Ollama (for the agent layer):**
```bash
# Install Ollama: https://ollama.com
ollama pull deepseek-r1:7b   # currently configured in src/agent/agents.py
# an earlier working prototype (src/agent/llm.ipynb) used qwen3.5:27b instead —
# not yet reconciled, see CLAUDE.md's open question
```

**Dataset:**
1. Join the [IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/rules).
2. Create a Kaggle API token → save to `~/.kaggle/kaggle.json`.
3. `python -m src.data.fetch_ieee_cis`

## Project layout

```
src/
  data/
    fetch_ieee_cis.py                 Kaggle download script
  features/
    feature_engineeringupdated.ipynb  THE current pipeline: load → clean → engineer
                                       → temporal split → encode → train → SHAP
                                       → similarity → agent, end to end
    utils.py                          Plotting / frequency-encoding helpers (imported)
    EDA_consolidated.ipynb            Consolidated EDA — informed drop/grouping decisions
    feature_engineeringupdated.ipynb  Includes the entire feature engineering, model training and agent usage now (to be split up!)
    README.md                         Column-by-column EDA breakdown
  model/
    explain.py                        SHAPExplainer — imported by the current notebook
  agent/
    agents.py                         Main agent + judge (investigate())
    tools.py                          SHAP / similar-fraud tools, ToolContext
    prompts.py                        System prompts for both subagents + judge
    README.md                         Local-model VRAM/benchmark notes
data/raw/                             gitignored — downloaded CSVs
```

## Current results

Full IEEE-CIS dataset, temporal 80/20 split, true 3.44%-fraud holdout
(`feature_engineeringupdated.ipynb`):

| Model | AUC | PR-AUC |
|---|---|---|
| LightGBM | 0.9110 | 0.5194 |
| XGBoost | 0.9050 | 0.5084 |
| CatBoost | 0.9079 | 0.4960 |
| Ensemble (rank-averaged) | 0.9138 | 0.5213 |

See CLAUDE.md for the full experiment history, including an earlier `.py`-pipeline
tuning pass that reached LightGBM AUPRC 0.5057 with tuned hyperparameters.

## Running tests

```bash
pytest tests/ -v
```

## Update conda environment

```bash
conda env export --from-history > environment.yml
```