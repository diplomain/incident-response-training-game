# IncidentIQ — Predict & Respond

A text-based public service incident response training game where players analyze synthetic, fictional civic-service incident event sequences, discover frequent event patterns using `thilimarketbasket`, predict the next likely event using statistical and neural language models, and allocate abstract response resources under a strict budget.

> [!NOTE]
> **Educational & Fictional Disclaimer**: All scenarios, events, resource costs, and requirements in IncidentIQ are completely synthetic and fictional. This project is an educational and analytical toy and must **not** be used for real emergency forecasting, dispatch, or operational decision-making.

---

## Features

- **Milestone 1 — Ordered Sequence Dataset**: Reproducible synthetic dataset stored in `data/incident_sequences.json`, loaded directly without duplicated hardcoding, and protected by strict schema validation.
- **Milestone 2 — Frequent Sequence Mining**: Direct integration with the required `thilimarketbasket.mine` package (`frequent_itemsets` and `rules`) to calculate exact support counts, percentages, and association rules, with a transparent pure-Python Apriori fallback.
- **Milestone 3 — Next-Event Prediction**: Context-aware n-gram sequence predictor evaluated against Random and Most Frequent Unigram baselines on a held-out test split without train/test data leakage.
- **Milestone 4 — Genuine Neural Language Model (CPU-Friendly)**: Pure-NumPy autoregressive neural language model (token embeddings + causal recurrent sequence encoder + softmax output head) trained locally via cross-entropy loss and Adam optimization. Runs locally in seconds without GPU, paid APIs, or external API keys.
- **Milestone 5 — Terminal Game Engine & Simulation**: Interactive terminal gameplay and automated simulation mode with resource budgeting, allocation coverage, over-allocation penalties, event reveals, and session score persistence.
- **Evaluation & Evidence**: Reproducible evaluation benchmark exporting Top-1, Top-3, MRR, and game efficiency metrics to JSON, CSV, gameplay logs, and Markdown reports.

---

## Setup Instructions

IncidentIQ requires Python 3.10+ (Python 3.12 is recommended).

### 1. Clone & Environment Setup

#### Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables (Optional)
Copy `.env.example` to `.env` if you wish to customize local storage paths:
```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

---

## CLI Commands

All commands are executed using the module entry point `python -m app.cli <command>`:

| Command | Description | Example |
|---|---|---|
| `hello` | Verify installation and pattern mining engine status | `python -m app.cli hello` |
| `generate-dataset` | Generate reproducible synthetic incident sequences | `python -m app.cli generate-dataset --count 120 --seed 42` |
| `patterns` | Mine frequent event patterns and association rules | `python -m app.cli patterns --support 0.20 --min-confidence 0.3` |
| `predict-demo` | Demonstrate next-event predictions from N-Gram & Neural LM | `python -m app.cli predict-demo --model both` |
| `train-lm` | Train the genuine CPU neural autoregressive language model | `python -m app.cli train-lm --epochs 30 --lr 0.015` |
| `evaluate` | Run reproducible benchmarks and output CSV/JSON/MD evidence | `python -m app.cli evaluate --seed 42 --epochs 30` |
| `play` | Start an interactive terminal training game session | `python -m app.cli play --rounds 5 --budget 100` |
| `play --auto` | Run an automated simulation session without interactive input | `python -m app.cli play --rounds 10 --budget 200 --auto` |
| `leaderboard` | View previous game session scores and history | `python -m app.cli leaderboard` |

---

## Detailed Milestone Verification

### Milestone 1 — Incident Sequence Dataset
- **Single Source of Truth**: All incident sequences are stored in `data/incident_sequences.json`. The application and game engine load directly from this file using `app.dataset.load_incidents()`.
- **Validation**: `app.dataset.validate_incidents()` validates that every sequence is a non-empty list of event names with length $\ge 2$, rejecting malformed structures, empty event strings, and invalid types.
- **Reproducible Generator**: `python -m app.cli generate-dataset` generates structured sequences across 8 civic-service domains (Weather, Power, Water, Roadway, Crowd Transit, Heat Advisory, Sanitation, Street Lighting) using fixed random seeds.

### Milestone 2 — Frequent Sequence Mining
- **`thilimarketbasket` Integration**: Calls `thilimarketbasket.mine.frequent_itemsets(baskets, min_support, max_size)` and `thilimarketbasket.mine.rules(itemsets, min_confidence)`.
- **Output Metrics**: Produces support counts ($k$), support shares ($k/N$), and association rules with confidence and lift.
- **Transparent Fallback**: If `thilimarketbasket` is unavailable in an environment, `app.mining.mine_frequent_patterns()` automatically executes a verified local Apriori fallback, explicitly reporting `is_fallback: True` and `engine: "fallback_apriori"` without misrepresenting the source.

### Milestone 3 — Next-Event Prediction
- **Leakage-Free Train/Test Split**: `app.prediction.train_test_split()` partitions sequences into an 80% training set and a 20% held-out test split using fixed random seeds before any model training occurs.
- **Baselines**: Evaluated against:
  1. *Random Baseline*: Uniform distribution across the 19 vocabulary events.
  2. *Most Frequent Unigram Baseline*: Historical unigram event frequencies in the training set.
- **Accuracy Metrics**: Evaluates **Top-1 Accuracy**, **Top-3 Accuracy**, and **Mean Reciprocal Rank (MRR)** across all prefix transitions in the held-out test set.

### Milestone 4 — Language Model Continued Pretraining
- **Genuine Neural Language Model**: Built in `app/neural_lm.py` (`NeuralNextEventLM`). Contains:
  - Token embedding layer ($V \to d_{emb}$).
  - Causal recurrent hidden state encoder with tanh activation.
  - Output projection head ($d_h \to V$) and softmax cross-entropy loss.
  - Backpropagation Through Time (BPTT) and Adam optimizer with gradient clipping.
- **Documentation of Training**:
  - *What is trained*: Embeddings, input-to-hidden weights, recurrent weights, output projection weights, biases, and Adam moment vectors.
  - *What is pretrained*: Initialized using Xavier standard normal initialization. No paid APIs, external tokens, or API keys are required.
  - *Loss Convergence*: On 96 training sequences, cross-entropy loss drops from $\sim 1.56$ to $< 0.70$ ($PPL \approx 2.01$) in 30 epochs on CPU in $< 8$ seconds.
- **Optional Transformer Pipeline**: `scripts/train_transformer_lm.py` provides an optional PyTorch/HuggingFace script for users wishing to continued-pretrain small transformers like `distilgpt2` or `SmolLM`.

### Milestone 5 — Game Engine & Scoring
- **Terminal Gameplay**: Players observe an evolving incident context (e.g. `Heavy_Rain -> Waterlogging`), review model suggestions, predict the next likely event, and allocate abstract resources (`teams`, `equipment`, `communications`).
- **Budget Enforced**: Every resource costs points (`teams`: 10, `equipment`: 5, `communications`: 5). Allocations exceeding the remaining budget are strictly rejected and re-prompted.
- **Scoring Details**:
  - *Prediction Accuracy*: Correct predictions / total rounds $\times 100\%$.
  - *Resource Coverage*: $\frac{\sum \min(\text{allocation}[r], \text{needs}[r])}{\sum \text{needs}[r]}$.
  - *Resource Efficiency*: Coverage minus over-allocation penalty ($0.05 \times \text{over-allocated units}$).
  - *Overall Score*: $0.6 \times \text{accuracy} + 0.4 \times \text{efficiency}$.

---

## Evaluation Benchmark & Evidence

Run the complete evaluation suite to reproduce the metrics:

```powershell
python -m app.cli evaluate --seed 42 --epochs 30 --output-dir evaluation
```

### Empirical Results (80/20 Held-Out Test Split, 70 Transitions)

| Model | Top-1 Accuracy (%) | Top-3 Accuracy (%) | MRR | Total Transitions |
|---|---|---|---|---|
| **Random Baseline** | 12.86% | 15.71% | 0.1429 | 70 |
| **Most Frequent (Unigram) Baseline** | 12.86% | 32.86% | 0.2119 | 70 |
| **3-Gram Predictor (Statistical)** | **70.00%** | 95.71% | **0.8286** | 70 |
| **Neural Autoregressive LM (CPU)** | 65.71% | **100.00%** | **0.8286** | 70 |

### Gameplay Simulation Benchmark (20 Rounds)

| Strategy | Prediction Accuracy | Resource Efficiency | Overall Score (60/40) | Total Spent |
|---|---|---|---|---|
| **Random Player** | 5.0% | 23.33% | 12.33 | 200 pts |
| **N-Gram Assisted Player** | 75.0% | 32.00% | 57.80 | 200 pts |
| **Neural LM Assisted Player** | **80.0%** | **33.00%** | **61.20** | 200 pts |

Generated evaluation evidence files in `evaluation/`:
- `evaluation/evaluation_report.md`: Markdown summary report.
- `evaluation/prediction_metrics.json`: Full machine-readable split info and metric results.
- `evaluation/prediction_metrics.csv`: Tabular prediction metrics.
- `evaluation/gameplay_evaluation.json`: Simulation benchmark scores.
- `evaluation/gameplay_log.txt`: Recorded terminal gameplay transcript.

---

## Dataset Limitations & Honest Evaluation Notes

1. **Synthetic Regularity**: The dataset reflects structured civic workflows. In real municipal operations, incidents feature unpredictable delays, unrecorded events, and ambiguous cross-agency labels.
2. **Vocabulary Size**: The domain vocabulary currently contains 19 discrete events. With compact vocabularies, statistical n-grams perform very strongly alongside neural models.
3. **Data Sparsity in Small Splits**: When evaluating small samples (e.g. 20-120 sequences), rare incident branches have few training examples; maintaining stratified sequence sampling is essential to avoid evaluation variance.

---

## Running the Automated Test Suite

Run all unit tests using pytest:

```powershell
pytest -v
```

All 49 unit tests pass across dataset loading, schema validation, `thilimarketbasket` mining, transparent fallback, n-gram predictions, neural LM gradient updates, resource budgeting, scoring, storage, and CLI subcommands.
