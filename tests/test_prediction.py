"""Unit tests for next-event prediction and baseline comparison (Milestone 3)."""
import pytest
from app.prediction import (
    train_test_split,
    RandomBaseline,
    MostFrequentBaseline,
    NGramPredictor,
    evaluate_predictor,
    milestone_3,
)


def test_train_test_split_no_leakage():
    data = [[f"Event_{i}_{j}" for j in range(3)] for i in range(20)]
    train, test = train_test_split(data, test_ratio=0.25, seed=42)
    assert len(train) == 15
    assert len(test) == 5
    # Ensure sets are disjoint by sequence identity
    train_tuples = {tuple(s) for s in train}
    test_tuples = {tuple(s) for s in test}
    assert train_tuples.isdisjoint(test_tuples)


def test_random_baseline():
    data = [["A", "B", "C"], ["B", "C", "D"]]
    model = RandomBaseline().train(data)
    preds = model.predict(["A"], top_k=3)
    assert len(preds) == 3
    # Uniform probabilities
    assert abs(preds[0][1] - 0.25) < 1e-6


def test_most_frequent_baseline():
    data = [["A", "B"], ["A", "C"], ["A", "D"]]
    model = MostFrequentBaseline().train(data)
    preds = model.predict(["B"], top_k=1)
    assert preds[0][0] == "A"


def test_ngram_predictor_probabilities_and_top_k():
    data = [["A", "B", "C"], ["A", "B", "D"], ["A", "B", "C"]]
    model = NGramPredictor().train(data)
    preds = model.predict(["A", "B"], top_k=2)
    assert len(preds) == 2
    assert preds[0][0] == "C"
    assert abs(preds[0][1] - (2 / 3)) < 1e-6
    assert abs(sum(p for _, p in preds) - 1.0) < 1e-6


def test_evaluate_predictor_metrics():
    train = [["A", "B", "C"], ["X", "Y", "Z"]]
    test = [["A", "B", "C"]]
    model = NGramPredictor().train(train)
    metrics = evaluate_predictor(model, test, top_k_values=(1, 3))
    assert metrics["total_transitions"] == 2  # A->B and [A,B]->C
    assert metrics["top_1_accuracy"] == 100.0
    assert metrics["top_3_accuracy"] == 100.0


def test_milestone_3_backwards_compatibility():
    model = milestone_3([["A", "B", "C"], ["A", "B", "D"]])
    predictions = model.predict(["A", "B"])
    assert predictions[0][0] in {"C", "D"}
    assert abs(sum(prob for _, prob in predictions) - 1) < 1e-9
