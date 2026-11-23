"""Automated tests for core IncidentIQ milestones."""
from app.core import milestone_1, milestone_2, milestone_3, milestone_5, resource_cost


def test_milestone_1_returns_valid_ordered_sequences():
    sample = [["A", "B", "C"], ["B", "C"]]
    assert milestone_1(sample) == sample
    assert milestone_1()[0][0] == "Heavy_Rain"


def test_milestone_1_rejects_invalid_reports():
    import pytest
    with pytest.raises(ValueError):
        milestone_1([["A", "", "B"]])


def test_milestone_2_mines_ordered_patterns_with_support():
    reports = [["A", "B", "C"], ["A", "B", "D"], ["A", "C"]]
    results = milestone_2(reports, min_support=0.66, max_pattern_length=2)
    assert {tuple(r["pattern"]) for r in results} >= {("A",), ("B",), ("A", "B")}
    assert next(r for r in results if r["pattern"] == ["A", "B"])["count"] == 2


def test_milestone_3_predicts_next_event_and_probabilities_sum():
    model = milestone_3([["A", "B", "C"], ["A", "B", "D"]])
    predictions = model.predict(["A", "B"])
    assert predictions[0][0] in {"C", "D"}
    assert abs(sum(prob for _, prob in predictions) - 1) < 1e-9


def test_resource_cost_calculation():
    assert resource_cost({"teams": 2, "equipment": 3, "communications": 1}) == 40


def test_milestone_5_evaluation_metrics():
    results = [
        {"correct": True, "efficiency": 0.8},
        {"correct": False, "efficiency": 0.6},
    ]
    scores = milestone_5(results)
    assert scores["prediction_accuracy"] == 50
    assert scores["resource_efficiency"] == 70
    assert scores["overall_score"] == 58
    assert scores["rounds"] == 2


def test_milestone_5_handles_empty_results():
    assert milestone_5([])["rounds"] == 0
