"""Unit tests for the IncidentIQ game engine, allocation, and scoring (Milestone 5)."""
import pytest
from app.game import (
    resource_cost,
    calculate_round_metrics,
    milestone_5,
    play_game,
)
from app.storage import Store


def test_resource_cost_calculation():
    assert resource_cost({"teams": 2, "equipment": 3, "communications": 1}) == (2 * 10 + 3 * 5 + 1 * 5)
    assert resource_cost({}) == 0
    assert resource_cost({"teams": -2}) == 0


def test_calculate_round_metrics_exact_match():
    # Waterlogging needs: teams: 2, equipment: 3, communications: 1 (sum = 6)
    alloc = {"teams": 2, "equipment": 3, "communications": 1}
    coverage, efficiency = calculate_round_metrics("Waterlogging", alloc)
    assert coverage == 1.0
    assert efficiency == 1.0


def test_calculate_round_metrics_overallocation_penalty():
    # Waterlogging needs: 2, 3, 1
    # Over-allocate: teams: 4 (+2), equipment: 4 (+1), comms: 2 (+1) -> over = 4
    alloc = {"teams": 4, "equipment": 4, "communications": 2}
    coverage, efficiency = calculate_round_metrics("Waterlogging", alloc)
    assert coverage == 1.0
    assert efficiency == 1.0 - 0.05 * 4  # 0.80


def test_calculate_round_metrics_underallocation():
    # Waterlogging needs: 2, 3, 1 (total 6)
    alloc = {"teams": 1, "equipment": 0, "communications": 0}
    coverage, efficiency = calculate_round_metrics("Waterlogging", alloc)
    assert abs(coverage - (1 / 6)) < 1e-6
    assert abs(efficiency - (1 / 6)) < 1e-6


def test_milestone_5_scoring():
    results = [
        {"correct": True, "efficiency": 0.8, "spent": 20},
        {"correct": False, "efficiency": 0.6, "spent": 15},
    ]
    scores = milestone_5(results)
    assert scores["prediction_accuracy"] == 50.0
    assert scores["resource_efficiency"] == 70.0
    assert scores["overall_score"] == 58.0
    assert scores["rounds"] == 2
    assert scores["total_spent"] == 35


def test_milestone_5_empty():
    assert milestone_5([])["rounds"] == 0


def test_play_game_simulation_mode(tmp_path):
    sample_reports = [
        ["Heavy_Rain", "Waterlogging", "Traffic_Disruption", "Cleanup"],
        ["Power_Outage", "Service_Disruption", "Repair_Team", "Repair_Complete"],
    ]
    res = play_game(
        rounds=3,
        budget=100,
        reports=sample_reports,
        interactive=False,
        simulated_strategy="model",
        seed=42,
        store_session=False,
    )
    assert len(res["round_records"]) == 3
    scores = res["session_scores"]
    assert scores["rounds"] == 3
    assert 0 <= scores["prediction_accuracy"] <= 100
    assert 0 <= scores["resource_efficiency"] <= 100
    assert 0 <= scores["remaining_budget"] <= 100
