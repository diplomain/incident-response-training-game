# IncidentIQ Evaluation Report

## 1. Overview & Setup
- **Dataset**: Synthetic civic-service incident sequences (120 sequences).
- **Split**: 80% Train (96 sequences, 272 transitions) / 20% Test (24 sequences, 70 transitions).
- **Leakage Prevention**: Strictly split by full incident sequences prior to training.
- **Random Seed**: 42.
- **Environment**: Local CPU execution without external APIs or paid services.

---

## 2. Next-Event Prediction Benchmark (Held-out Test Split)

| Model | Top-1 Accuracy (%) | Top-3 Accuracy (%) | MRR | Total Transitions |
|---|---|---|---|---|
| Random Baseline | 12.86% | 15.71% | 0.1429 | 70 |
| Most Frequent (Unigram) Baseline | 12.86% | 32.86% | 0.2119 | 70 |
| 3-Gram Predictor (Statistical) | 70.0% | 95.71% | 0.8286 | 70 |
| Neural Autoregressive LM (CPU) | 65.71% | 100.0% | 0.8286 | 70 |

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
- **Training Epochs**: 30 epochs trained in 7.06s on CPU.
- **Loss Objective**: Causal Next-Token Cross-Entropy Loss.
- **Initial Loss -> Final Loss**: 1.5597 -> 0.6981
- **Perplexity**: 2.01

---

## 4. Game Engine Simulation Benchmark (Milestone 5)

| Player Strategy | Prediction Accuracy (%) | Resource Efficiency (%) | Overall Score (60/40) | Total Spent (pts) |
|---|---|---|---|---|
| Random Player | 5.0% | 23.33% | 12.33 | 200 |
| N-Gram Assisted | 75.0% | 32.0% | 57.8 | 200 |
| Neural LM Assisted | 80.0% | 33.0% | 61.2 | 200 |

---

## 5. Dataset Limitations & Honest Evaluation Notes
1. **Synthetic Nature**: Sequences represent educational, civic-service scenarios with clear canonical motifs rather than chaotic real-world incidents.
2. **Vocabulary Size**: The current event vocabulary contains 19 discrete events. With a small vocabulary, statistical n-grams perform very competitively alongside neural models.
3. **Sparsity**: In small datasets (20-120 sequences), rare sequence branches have few training examples; held-out splits must maintain balanced representations to prevent high evaluation variance.
