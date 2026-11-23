"""Core module for IncidentIQ.

Central facade integrating the upgraded dataset loading, thilimarketbasket pattern mining,
next-event prediction, neural language model, and game engine.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .dataset import (
    CANONICAL_EVENTS,
    load_incidents,
    validate_incidents,
    generate_synthetic_incidents,
    save_incidents,
)
from .mining import (
    THILI_AVAILABLE,
    mine_frequent_patterns,
    milestone_2,
)
from .prediction import (
    NGramPredictor,
    RandomBaseline,
    MostFrequentBaseline,
    train_test_split,
    evaluate_predictor,
    milestone_3,
)
from .neural_lm import (
    NeuralNextEventLM,
    train_neural_lm,
)
from .game import (
    RESOURCE_TYPES,
    RESOURCE_COSTS,
    RESOURCE_NEEDS,
    resource_cost,
    calculate_round_metrics,
    play_game,
    milestone_4,
    milestone_5,
)

# Dynamic load of sequences from data file (avoids duplication)
try:
    INCIDENT_SEQUENCES = load_incidents()
except Exception:
    # Safe fallback if file path resolution differs during early setup
    INCIDENT_SEQUENCES = generate_synthetic_incidents(count=20, seed=42)


# Backwards compatibility class alias for Milestone 3
class NextEventModel(NGramPredictor):
    """Backwards-compatible alias for NGramPredictor."""
    def __init__(self, reports: Optional[Sequence[Sequence[str]]] = None):
        super().__init__(max_n=3)
        self.reports = milestone_1(reports)

    def train(self) -> "NextEventModel":
        super().train(self.reports)
        return self


# Milestone 1 facade
def milestone_1(reports: Optional[Sequence[Sequence[str]]] = None) -> List[List[str]]:
    """Validate and return ordered reports. Loads from data file if None."""
    if reports is None:
        return load_incidents()
    return validate_incidents(reports, allow_unknown=True)


__all__ = [
    "CANONICAL_EVENTS",
    "INCIDENT_SEQUENCES",
    "RESOURCE_TYPES",
    "RESOURCE_COSTS",
    "RESOURCE_NEEDS",
    "NextEventModel",
    "NGramPredictor",
    "NeuralNextEventLM",
    "RandomBaseline",
    "MostFrequentBaseline",
    "load_incidents",
    "validate_incidents",
    "generate_synthetic_incidents",
    "save_incidents",
    "mine_frequent_patterns",
    "train_test_split",
    "evaluate_predictor",
    "train_neural_lm",
    "resource_cost",
    "calculate_round_metrics",
    "play_game",
    "milestone_1",
    "milestone_2",
    "milestone_3",
    "milestone_4",
    "milestone_5",
]
