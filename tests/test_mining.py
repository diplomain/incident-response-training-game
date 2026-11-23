"""Unit tests for frequent pattern mining and thilimarketbasket integration (Milestone 2)."""
import pytest
from app.mining import (
    THILI_AVAILABLE,
    mine_frequent_patterns,
    milestone_2,
    _local_apriori,
    _local_rules,
)


def test_thili_is_available():
    # Verify thilimarketbasket package is installed and importable
    assert THILI_AVAILABLE is True, "thilimarketbasket should be installed in the environment"


def test_mine_patterns_with_thili():
    reports = [
        ["Heavy_Rain", "Waterlogging", "Cleanup"],
        ["Heavy_Rain", "Waterlogging", "Drainage_Check"],
        ["Heavy_Rain", "Road_Blockage", "Cleanup"],
        ["Power_Outage", "Service_Disruption"],
    ]
    res = mine_frequent_patterns(reports, min_support=0.5, max_size=2)
    assert res["engine"] == "thilimarketbasket"
    assert res["is_fallback"] is False
    assert res["total_sequences"] == 4
    # Heavy_Rain appears in 3/4 = 75%
    rain_pattern = next(p for p in res["patterns"] if p["pattern"] == ["Heavy_Rain"])
    assert rain_pattern["support"] == 0.75
    assert rain_pattern["count"] == 3


def test_mine_patterns_rules():
    reports = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "B", "D"],
        ["E", "F"],
    ]
    res = mine_frequent_patterns(reports, min_support=0.5, max_size=2, min_confidence=0.5)
    assert len(res["rules"]) > 0
    rule = next(r for r in res["rules"] if r["antecedent"] == ["A"] and r["consequent"] == ["B"])
    assert rule["confidence"] == 1.0


def test_transparent_fallback():
    reports = [
        ["A", "B", "C"],
        ["A", "B", "D"],
        ["A", "C"],
    ]
    res = mine_frequent_patterns(reports, min_support=0.66, max_size=2, force_fallback=True)
    assert res["engine"] == "fallback_apriori"
    assert res["is_fallback"] is True
    assert any(p["pattern"] == ["A", "B"] for p in res["patterns"])


def test_milestone_2_compatibility():
    reports = [["A", "B", "C"], ["A", "B", "D"], ["A", "C"]]
    results = milestone_2(reports, min_support=0.66, max_pattern_length=2)
    patterns = {tuple(r["pattern"]) for r in results}
    assert ("A",) in patterns or ("A", "B") in patterns
    ab_entry = next(r for r in results if r["pattern"] == ["A", "B"])
    assert ab_entry["count"] == 2
