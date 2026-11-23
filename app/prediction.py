"""Next-Event Predictors and Evaluation for IncidentIQ.

Implements statistical sequence predictors, baselines, train/test split,
and evaluation for top-1 and top-3 accuracy on held-out transitions without data leakage.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .dataset import load_incidents, validate_incidents


def train_test_split(
    sequences: Sequence[Sequence[str]],
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Split sequences into train and held-out test sets without data leakage.

    Args:
        sequences: The incident sequences.
        test_ratio: Proportion of sequences held out for testing.
        seed: Random seed for deterministic reproducibility.

    Returns:
        (train_sequences, test_sequences)
    """
    valid_seqs = [list(seq) for seq in sequences]
    rng = random.Random(seed)
    shuffled = list(valid_seqs)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_test = max(1, int(round(n_total * test_ratio)))
    n_train = n_total - n_test

    train_seqs = shuffled[:n_train]
    test_seqs = shuffled[n_train:]
    return train_seqs, test_seqs


class RandomBaseline:
    """Baseline predictor: uniform random probability across known training vocabulary."""

    def __init__(self):
        self.vocab: List[str] = []
        self.name = "Random Baseline"

    def train(self, sequences: Sequence[Sequence[str]]) -> "RandomBaseline":
        vocab_set = set()
        for seq in sequences:
            vocab_set.update(seq)
        self.vocab = sorted(list(vocab_set))
        return self

    def predict(self, context: Sequence[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.vocab:
            return []
        prob = 1.0 / len(self.vocab)
        return [(event, prob) for event in self.vocab[:top_k]]


class MostFrequentBaseline:
    """Baseline predictor: predicts most common unigram events in the training corpus."""

    def __init__(self):
        self.unigram_counts: Counter = Counter()
        self.total_tokens: int = 0
        self.name = "Most Frequent (Unigram) Baseline"

    def train(self, sequences: Sequence[Sequence[str]]) -> "MostFrequentBaseline":
        self.unigram_counts.clear()
        self.total_tokens = 0
        for seq in sequences:
            for event in seq:
                self.unigram_counts[event] += 1
                self.total_tokens += 1
        return self

    def predict(self, context: Sequence[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if self.total_tokens == 0:
            return []
        top_items = self.unigram_counts.most_common(top_k)
        total_top = sum(cnt for _, cnt in top_items) or 1
        return [(event, cnt / self.total_tokens) for event, cnt in top_items]


class NGramPredictor:
    """Sequential next-event model using n-gram context counts with backoff."""

    def __init__(self, max_n: int = 3):
        self.max_n = max_n
        self.transitions: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.unigram: Counter = Counter()
        self.total_unigrams: int = 0
        self.name = f"{max_n}-Gram Predictor (Statistical)"
        self.trained = False

    def train(self, sequences: Sequence[Sequence[str]]) -> "NGramPredictor":
        self.transitions.clear()
        self.unigram.clear()
        self.total_unigrams = 0

        for seq in sequences:
            for i, event in enumerate(seq):
                self.unigram[event] += 1
                self.total_unigrams += 1
                for n in range(1, self.max_n + 1):
                    if i >= n:
                        context = tuple(seq[i - n : i])
                        self.transitions[context][event] += 1

        self.trained = True
        return self

    def predict(self, context: Sequence[str], top_k: int = 3) -> List[Tuple[str, float]]:
        if not self.trained:
            raise RuntimeError("Model must be trained before predicting.")

        ctx = tuple(context or [])
        # Try contexts from max_n down to 1
        for n in range(min(self.max_n, len(ctx)), 0, -1):
            key = ctx[-n:]
            if key in self.transitions and self.transitions[key]:
                counts = self.transitions[key]
                total = sum(counts.values())
                return [(event, count / total) for event, count in counts.most_common(top_k)]

        # Fallback to unigram popularity
        if self.total_unigrams > 0:
            return [(event, count / self.total_unigrams) for event, count in self.unigram.most_common(top_k)]
        return []


def evaluate_predictor(
    predictor: Any,
    test_sequences: Sequence[Sequence[str]],
    top_k_values: Tuple[int, ...] = (1, 3),
) -> Dict[str, Any]:
    """Evaluate a predictor on held-out test sequences across all prefix transitions.

    Args:
        predictor: Model with a .predict(context, top_k) method.
        test_sequences: Held-out sequence dataset.
        top_k_values: Accuracies to compute (default: top-1, top-3).

    Returns:
        Dict with total transitions, top-1 accuracy, top-3 accuracy, and MRR.
    """
    total_evals = 0
    top_1_hits = 0
    top_3_hits = 0
    reciprocal_ranks = []

    for seq in test_sequences:
        for i in range(1, len(seq)):
            context = seq[:i]
            target = seq[i]
            total_evals += 1

            preds = predictor.predict(context, top_k=max(top_k_values))
            pred_events = [event for event, _ in preds]

            if pred_events and pred_events[0] == target:
                top_1_hits += 1

            if target in pred_events[:3]:
                top_3_hits += 1

            if target in pred_events:
                rank = pred_events.index(target) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

    top_1_acc = (top_1_hits / total_evals) if total_evals > 0 else 0.0
    top_3_acc = (top_3_hits / total_evals) if total_evals > 0 else 0.0
    mrr = (sum(reciprocal_ranks) / total_evals) if total_evals > 0 else 0.0

    return {
        "model_name": getattr(predictor, "name", type(predictor).__name__),
        "total_transitions": total_evals,
        "top_1_accuracy": round(top_1_acc * 100, 2),
        "top_3_accuracy": round(top_3_acc * 100, 2),
        "mrr": round(mrr, 4),
    }


def milestone_3(reports: Optional[Sequence[Sequence[str]]] = None) -> NGramPredictor:
    """Milestone 3 interface: train and return the standard next-event statistical model."""
    sequences = load_incidents() if reports is None else validate_incidents(reports)
    model = NGramPredictor(max_n=3)
    model.train(sequences)
    return model
