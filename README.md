# IncidentIQ — Predict & Respond

A text-based training game using **fictional civic-service incident sequences**. Players predict the next event and allocate abstract response-resource points under a budget. It is an educational toy and not a real emergency forecasting or dispatch tool.

## Features
- Ordered event-sequence dataset (JSON)
- Frequent ordered subsequence discovery with support counts
- Next-event model (context n-gram counts; local and key-free)
- Terminal gameplay with prediction, resource allocation, budget limits, and feedback
- Scoring for prediction accuracy, resource efficiency, and a weighted overall score
- Tests for all implemented core components

## Setup
Use Python 3.10+ (Python 3.12 is recommended):

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
python -m app.cli hello
python -m app.cli patterns --support 0.20
python -m app.cli predict-demo
python -m app.cli play --rounds 5 --budget 100
pytest -q
```

## Milestones
- [x] **1. Ordered incident sequences:** validate and use reports as lists of ordered events.
- [x] **2. Frequent patterns:** mine ordered subsequences and calculate support. This prototype uses a small transparent implementation; the course-specific `thilimarketbasket` package remains in dependencies, but its public API was not confirmed and is not yet called by the code.
- [x] **3. Next-event LM baseline:** train a context n-gram frequency model and output probability-ranked next events. This is a lightweight local baseline, **not neural continued pretraining**. True continued pretraining would require a supported model, tokenizer, training dataset pipeline, and training compute; it is a future extension.
- [x] **4. Text game:** rounds, player prediction, resource allocations, budget checks, reveal, and feedback.
- [x] **5. Evaluation:** prediction accuracy, average resource efficiency, and weighted score (60% accuracy + 40% efficiency).

## Scoring details
- Prediction accuracy = correct predictions / rounds × 100.
- Resource coverage compares allocation quantities with the fictional event's resource requirement table.
- Round resource efficiency = coverage minus a modest over-allocation penalty (bounded 0–100%).
- Overall score = 60% prediction accuracy + 40% average resource efficiency.

All resource needs and costs are arbitrary game values, not real-world operational recommendations.


## Stretch ideas
- Add more abstract resource types and persistent budgets.
- Add difficulty modes that adjust how much sequence patterns are repeated.
- Reveal player prediction and model prediction side-by-side after each round.
- Save results to JSON and add a leaderboard.
- Add an optional external LM for narrative hints, while keeping predictions grounded in the trained dataset.

## Dataset
`data/incident_sequences.json` contains synthetic, fictional sequences. Avoid interpreting the game's patterns as real-world incident forecasts.
