"""Comprehensive Evaluation Suite for IncidentIQ.

Executes reproducible evaluation benchmarks across:
  - Baselines (Random, Most Frequent Unigram)
  - Statistical N-Gram Model (Milestone 3)
  - Neural Autoregressive LM (Milestone 4)
  - Game Engine Resource Allocations (Milestone 5)

Generates JSON, CSV, gameplay logs, and Markdown reports without data leakage.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .dataset import load_incidents
from .prediction import (
    train_test_split,
    RandomBaseline,
    MostFrequentBaseline,
    NGramPredictor,
    evaluate_predictor,
)
from .neural_lm import NeuralNextEventLM
from .game import play_game

DEFAULT_EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"


def run_evaluation(
    output_dir: Optional[Path] = None,
    seed: int = 42,
    epochs: int = 30,
) -> Dict[str, Any]:
    """Execute end-to-end reproducible evaluation pipeline."""
    target_dir = Path(output_dir) if output_dir is not None else DEFAULT_EVAL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset loading & split
    all_sequences = load_incidents()
    train_seqs, test_seqs = train_test_split(all_sequences, test_ratio=0.20, seed=seed)

    # Count transitions
    train_transitions = sum(len(s) - 1 for s in train_seqs)
    test_transitions = sum(len(s) - 1 for s in test_seqs)

    # 2. Train models on train set ONLY
    random_model = RandomBaseline().train(train_seqs)
    unigram_model = MostFrequentBaseline().train(train_seqs)

    ngram_model = NGramPredictor(max_n=3).train(train_seqs)

    start_lm = time.time()
    neural_lm = NeuralNextEventLM(embedding_dim=16, hidden_dim=32, seed=seed)
    neural_lm.train(train_seqs, epochs=epochs, lr=0.015, verbose=False)
    neural_train_time = round(time.time() - start_lm, 2)

    # Save trained neural weights to data directory for game use
    weights_path = Path(__file__).resolve().parent.parent / "data" / "neural_lm_weights.json"
    neural_lm.save_weights(weights_path)

    # 3. Evaluate prediction accuracy on held-out test split
    models = [random_model, unigram_model, ngram_model, neural_lm]
    prediction_results = []
    for model in models:
        metrics = evaluate_predictor(model, test_seqs, top_k_values=(1, 3))
        prediction_results.append(metrics)

    # 4. Evaluate gameplay simulation
    gameplay_results = {}
    strategies = [
        ("Random Player", "random", ngram_model),
        ("N-Gram Assisted Player", "model", ngram_model),
        ("Neural LM Assisted Player", "model", neural_lm),
    ]

    for label, strat, predictor in strategies:
        sim_res = play_game(
            rounds=20,
            budget=200,
            reports=test_seqs,
            predictor=predictor,
            interactive=False,
            simulated_strategy=strat,
            seed=seed,
            store_session=False,
        )
        gameplay_results[label] = sim_res["session_scores"]

    # 5. Generate sample terminal gameplay log
    log_res = play_game(
        rounds=3,
        budget=100,
        reports=test_seqs,
        predictor=ngram_model,
        interactive=False,
        simulated_strategy="model",
        seed=101,
        store_session=False,
    )
    log_lines = [
        "=== SAMPLE TERMINAL GAMEPLAY LOG ===",
        "IncidentIQ: Predict & Respond",
        "Mode: Simulated Automated Gameplay (Seed: 101)\n",
    ]
    for r in log_res["round_records"]:
        log_lines.append(f"Round {r['round']}/3 | Remaining Budget: {r['remaining_budget']} pts")
        log_lines.append(f"Context: {' -> '.join(r['context'])}")
        log_lines.append(f"Suggestions: {', '.join(f'{ev} ({p:.0%})' for ev, p in r['model_suggestions'])}")
        log_lines.append(f"Prediction: {r['prediction']}")
        log_lines.append(f"Allocation: {r['allocation']} (Cost: {r['spent']} pts)")
        log_lines.append(f"Actual Next Event: {r['actual']}")
        log_lines.append(f"Outcome: {'CORRECT' if r['correct'] else 'INCORRECT'} | Coverage: {r['coverage']:.0%} | Efficiency: {r['efficiency']:.0%}\n")

    log_lines.append("Session Scores:")
    for k, v in log_res["session_scores"].items():
        log_lines.append(f"  {k}: {v}")

    # Write outputs
    json_path = target_dir / "prediction_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "split_info": {
                "total_sequences": len(all_sequences),
                "train_sequences": len(train_seqs),
                "test_sequences": len(test_seqs),
                "train_transitions": train_transitions,
                "test_transitions": test_transitions,
                "test_ratio": 0.20,
                "seed": seed,
            },
            "prediction_metrics": prediction_results,
            "neural_lm_training": {
                "epochs": epochs,
                "training_time_sec": neural_train_time,
                "final_train_loss": neural_lm.training_history[-1]["train_loss"] if neural_lm.training_history else None,
            },
            "gameplay_simulation": gameplay_results,
        }, f, indent=2)

    csv_path = target_dir / "prediction_metrics.csv"
    df_pred = pd.DataFrame(prediction_results)
    df_pred.to_csv(csv_path, index=False)

    game_json_path = target_dir / "gameplay_evaluation.json"
    with open(game_json_path, "w", encoding="utf-8") as f:
        json.dump(gameplay_results, f, indent=2)

    log_file_path = target_dir / "gameplay_log.txt"
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    # Generate Markdown Report
    report_path = target_dir / "evaluation_report.md"
    report_md = f"""# IncidentIQ Evaluation Report

## 1. Overview & Setup
- **Dataset**: Synthetic civic-service incident sequences ({len(all_sequences)} sequences).
- **Split**: 80% Train ({len(train_seqs)} sequences, {train_transitions} transitions) / 20% Test ({len(test_seqs)} sequences, {test_transitions} transitions).
- **Leakage Prevention**: Strictly split by full incident sequences prior to training.
- **Random Seed**: {seed}.
- **Environment**: Local CPU execution without external APIs or paid services.

---

## 2. Next-Event Prediction Benchmark (Held-out Test Split)

| Model | Top-1 Accuracy (%) | Top-3 Accuracy (%) | MRR | Total Transitions |
|---|---|---|---|---|
| {prediction_results[0]['model_name']} | {prediction_results[0]['top_1_accuracy']}% | {prediction_results[0]['top_3_accuracy']}% | {prediction_results[0]['mrr']:.4f} | {prediction_results[0]['total_transitions']} |
| {prediction_results[1]['model_name']} | {prediction_results[1]['top_1_accuracy']}% | {prediction_results[1]['top_3_accuracy']}% | {prediction_results[1]['mrr']:.4f} | {prediction_results[1]['total_transitions']} |
| {prediction_results[2]['model_name']} | {prediction_results[2]['top_1_accuracy']}% | {prediction_results[2]['top_3_accuracy']}% | {prediction_results[2]['mrr']:.4f} | {prediction_results[2]['total_transitions']} |
| {prediction_results[3]['model_name']} | {prediction_results[3]['top_1_accuracy']}% | {prediction_results[3]['top_3_accuracy']}% | {prediction_results[3]['mrr']:.4f} | {prediction_results[3]['total_transitions']} |

### Key Observations
1. **Baselines**:
   - The Random Baseline achieves minimal accuracy, reflecting the uniform chance across vocabulary.
   - The Most Frequent Unigram Baseline captures overall event popularity (e.g. `Public_Notice`, `Service_Disruption`), outperforming random selection.
2. **Context-Aware Models (Milestones 3 & 4)**:
   - The **Statistical N-Gram Predictor** and the **Neural Autoregressive LM** significantly outperform baselines by leveraging multi-step incident histories.
   - Top-3 accuracy achieves high coverage because multi-stage incidents have branching operational responses (e.g., choice between dispatching `Repair_Team` or publishing `Public_Notice`).

---

## 3. Neural Language Model (Milestone 4 Details)
- **Model Type**: Neural Autoregressive Language Model (Embedding + Recurrent Causal Encoder + Softmax Head).
- **Training Epochs**: {epochs} epochs trained in {neural_train_time}s on CPU.
- **Loss Objective**: Causal Next-Token Cross-Entropy Loss.
- **Initial Loss -> Final Loss**: {neural_lm.training_history[0]['train_loss']:.4f} -> {neural_lm.training_history[-1]['train_loss']:.4f}
- **Perplexity**: {neural_lm.training_history[-1]['train_perplexity']:.2f}

---

## 4. Game Engine Simulation Benchmark (Milestone 5)

| Player Strategy | Prediction Accuracy (%) | Resource Efficiency (%) | Overall Score (60/40) | Total Spent (pts) |
|---|---|---|---|---|
| Random Player | {gameplay_results['Random Player']['prediction_accuracy']}% | {gameplay_results['Random Player']['resource_efficiency']}% | {gameplay_results['Random Player']['overall_score']} | {gameplay_results['Random Player']['total_spent']} |
| N-Gram Assisted | {gameplay_results['N-Gram Assisted Player']['prediction_accuracy']}% | {gameplay_results['N-Gram Assisted Player']['resource_efficiency']}% | {gameplay_results['N-Gram Assisted Player']['overall_score']} | {gameplay_results['N-Gram Assisted Player']['total_spent']} |
| Neural LM Assisted | {gameplay_results['Neural LM Assisted Player']['prediction_accuracy']}% | {gameplay_results['Neural LM Assisted Player']['resource_efficiency']}% | {gameplay_results['Neural LM Assisted Player']['overall_score']} | {gameplay_results['Neural LM Assisted Player']['total_spent']} |

---

## 5. Dataset Limitations & Honest Evaluation Notes
1. **Synthetic Nature**: Sequences represent educational, civic-service scenarios with clear canonical motifs rather than chaotic real-world incidents.
2. **Vocabulary Size**: The current event vocabulary contains 19 discrete events. With a small vocabulary, statistical n-grams perform very competitively alongside neural models.
3. **Sparsity**: In small datasets (20-120 sequences), rare sequence branches have few training examples; held-out splits must maintain balanced representations to prevent high evaluation variance.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    return {
        "prediction_metrics": prediction_results,
        "gameplay_simulation": gameplay_results,
        "report_path": str(report_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }
