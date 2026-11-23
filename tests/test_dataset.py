"""Unit tests for dataset loading, validation, and generation (Milestone 1)."""
import pytest
from pathlib import Path
from app.dataset import (
    CANONICAL_EVENTS,
    load_incidents,
    validate_incidents,
    generate_synthetic_incidents,
    save_incidents,
)


def test_validate_incidents_success():
    sample = [["Heavy_Rain", "Waterlogging", "Cleanup"], ["Power_Outage", "Repair_Complete"]]
    validated = validate_incidents(sample)
    assert validated == sample


def test_validate_incidents_rejects_none():
    with pytest.raises(ValueError, match="cannot be None"):
        validate_incidents(None)


def test_validate_incidents_rejects_empty_event():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        validate_incidents([["Heavy_Rain", "  ", "Cleanup"]])


def test_validate_incidents_rejects_short_sequence():
    with pytest.raises(ValueError, match="at least 2 events"):
        validate_incidents([["Heavy_Rain"]])


def test_validate_incidents_rejects_non_sequence():
    with pytest.raises(ValueError, match="must be a sequence"):
        validate_incidents(["not_a_sequence"])


def test_validate_incidents_canonical_enforcement():
    with pytest.raises(ValueError, match="not in CANONICAL_EVENTS"):
        validate_incidents([["Alien_Invasion", "Cleanup"]], allow_unknown=False)


def test_load_incidents_default_file():
    incidents = load_incidents()
    assert len(incidents) >= 20
    assert incidents[0][0] == "Heavy_Rain"
    for seq in incidents:
        assert len(seq) >= 2
        for event in seq:
            assert event in CANONICAL_EVENTS


def test_generate_synthetic_incidents_reproducibility():
    seqs_1 = generate_synthetic_incidents(count=50, seed=123)
    seqs_2 = generate_synthetic_incidents(count=50, seed=123)
    assert seqs_1 == seqs_2
    assert len(seqs_1) == 50


def test_save_and_reload_incidents(tmp_path: Path):
    target = tmp_path / "test_incidents.json"
    sample = [["Heavy_Rain", "Waterlogging"], ["Power_Outage", "Repair_Complete"]]
    save_incidents(sample, target)
    reloaded = load_incidents(target)
    assert reloaded == sample
