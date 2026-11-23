"""Incident Sequence Dataset Management for IncidentIQ.

Provides dataset loading, validation, generation, and storage for fictional
civic-service training sequences.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Iterable, List, Sequence, Union

# Fictional civic-service event types recognized by the game
CANONICAL_EVENTS = frozenset({
    "Heavy_Rain",
    "Waterlogging",
    "Road_Blockage",
    "Traffic_Disruption",
    "Public_Notice",
    "Cleanup",
    "Drainage_Check",
    "Power_Outage",
    "Service_Disruption",
    "Repair_Team",
    "Repair_Complete",
    "Water_Supply_Issue",
    "Public_Event_Crowding",
    "Transit_Delay",
    "Support_Team",
    "Service_Restored",
    "Heat_Advisory",
    "Waste_Collection_Delay",
    "Street_Light_Issue",
})

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "incident_sequences.json"


def validate_incidents(
    reports: Iterable[Sequence[str]],
    allow_unknown: bool = True,
    min_length: int = 2,
) -> List[List[str]]:
    """Validate incident reports as ordered sequences of non-empty event names.

    Args:
        reports: An iterable of sequences containing event name strings.
        allow_unknown: If False, enforces all event names must belong to CANONICAL_EVENTS.
        min_length: Minimum number of events required per sequence (default: 2).

    Returns:
        A list of validated incident sequences (each a list of strings).

    Raises:
        ValueError: If reports is not a collection, or any report is invalid.
    """
    if reports is None:
        raise ValueError("Reports collection cannot be None.")
    if not isinstance(reports, (list, tuple)):
        # Convert iterable to list if it's not a list/tuple
        try:
            reports = list(reports)
        except TypeError as err:
            raise ValueError("Reports must be an iterable of sequences.") from err

    validated: List[List[str]] = []
    for idx, report in enumerate(reports):
        if not isinstance(report, (list, tuple)):
            raise ValueError(
                f"Report #{idx} must be a sequence (list or tuple), got {type(report).__name__}."
            )
        if len(report) < min_length:
            raise ValueError(
                f"Report #{idx} has length {len(report)}; must contain at least {min_length} events."
            )
        cleaned_sequence: List[str] = []
        for event_idx, event in enumerate(report):
            if not isinstance(event, str) or not event.strip():
                raise ValueError(
                    f"Report #{idx}, event #{event_idx} must be a non-empty string."
                )
            event_name = event.strip()
            if not allow_unknown and event_name not in CANONICAL_EVENTS:
                raise ValueError(
                    f"Report #{idx}, event '{event_name}' is not in CANONICAL_EVENTS."
                )
            cleaned_sequence.append(event_name)
        validated.append(cleaned_sequence)

    return validated


def load_incidents(path: Union[str, Path, None] = None) -> List[List[str]]:
    """Load and validate incident sequences from a JSON file.

    Args:
        path: File path to the JSON dataset. Defaults to data/incident_sequences.json.

    Returns:
        List of validated incident sequences.
    """
    target_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Incident dataset file not found at: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return validate_incidents(data)


def save_incidents(incidents: List[List[str]], path: Union[str, Path, None] = None) -> Path:
    """Validate and write incident sequences to a JSON file.

    Args:
        incidents: The sequences to save.
        path: File path to write. Defaults to data/incident_sequences.json.

    Returns:
        Path of the saved file.
    """
    target_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    validated = validate_incidents(incidents)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(validated, f, indent=2)
    return target_path


# Domain incident flow templates (seed blueprints for synthetic generation)
INCIDENT_TEMPLATES = [
    # Storm / flooding incidents
    {
        "category": "Weather",
        "stages": [
            ["Heavy_Rain"],
            ["Waterlogging", "Road_Blockage"],
            ["Traffic_Disruption", "Public_Notice"],
            ["Cleanup", "Drainage_Check"],
        ],
        "weight": 25,
    },
    # Power grid failures
    {
        "category": "Power",
        "stages": [
            ["Power_Outage"],
            ["Service_Disruption"],
            ["Public_Notice", "Repair_Team"],
            ["Repair_Team", "Repair_Complete"],
        ],
        "weight": 18,
    },
    # Water utility issues
    {
        "category": "Water",
        "stages": [
            ["Water_Supply_Issue"],
            ["Service_Disruption"],
            ["Public_Notice", "Repair_Team"],
            ["Repair_Complete"],
        ],
        "weight": 15,
    },
    # Road obstruction / traffic incidents
    {
        "category": "Road",
        "stages": [
            ["Road_Blockage"],
            ["Traffic_Disruption"],
            ["Public_Notice", "Repair_Team"],
            ["Cleanup"],
        ],
        "weight": 15,
    },
    # Public event / transit surge
    {
        "category": "Transit",
        "stages": [
            ["Public_Event_Crowding"],
            ["Transit_Delay"],
            ["Public_Notice", "Support_Team"],
            ["Support_Team", "Service_Restored"],
        ],
        "weight": 12,
    },
    # Heat wave
    {
        "category": "Heat",
        "stages": [
            ["Heat_Advisory"],
            ["Public_Notice", "Service_Disruption"],
            ["Support_Team"],
            ["Service_Restored"],
        ],
        "weight": 5,
    },
    # Sanitation / waste delay
    {
        "category": "Sanitation",
        "stages": [
            ["Waste_Collection_Delay"],
            ["Public_Notice", "Service_Disruption"],
            ["Cleanup", "Support_Team"],
            ["Service_Restored"],
        ],
        "weight": 5,
    },
    # Street lighting
    {
        "category": "Lighting",
        "stages": [
            ["Street_Light_Issue"],
            ["Service_Disruption"],
            ["Repair_Team"],
            ["Repair_Complete"],
        ],
        "weight": 5,
    },
]


def generate_synthetic_incidents(
    count: int = 120,
    seed: int = 42,
    include_canonical_seed: bool = True,
) -> List[List[str]]:
    """Generate a reproducible, domain-grounded synthetic dataset of incident sequences.

    Args:
        count: Total number of sequences to generate.
        seed: Random seed for exact reproducibility.
        include_canonical_seed: Whether to include the base 20 canonical sequences.

    Returns:
        List of generated ordered event sequences.
    """
    rng = random.Random(seed)
    sequences: List[List[str]] = []

    if include_canonical_seed:
        canonical_20 = [
            ["Heavy_Rain", "Waterlogging", "Traffic_Disruption", "Cleanup"],
            ["Heavy_Rain", "Waterlogging", "Traffic_Disruption", "Drainage_Check"],
            ["Heavy_Rain", "Road_Blockage", "Traffic_Disruption", "Cleanup"],
            ["Heavy_Rain", "Waterlogging", "Public_Notice", "Cleanup"],
            ["Heavy_Rain", "Waterlogging", "Traffic_Disruption", "Drainage_Check"],
            ["Power_Outage", "Service_Disruption", "Public_Notice", "Repair_Complete"],
            ["Power_Outage", "Service_Disruption", "Repair_Team", "Repair_Complete"],
            ["Power_Outage", "Service_Disruption", "Public_Notice", "Repair_Team"],
            ["Water_Supply_Issue", "Service_Disruption", "Public_Notice", "Repair_Complete"],
            ["Water_Supply_Issue", "Service_Disruption", "Repair_Team", "Repair_Complete"],
            ["Road_Blockage", "Traffic_Disruption", "Public_Notice", "Cleanup"],
            ["Road_Blockage", "Traffic_Disruption", "Repair_Team", "Cleanup"],
            ["Public_Event_Crowding", "Transit_Delay", "Public_Notice", "Service_Restored"],
            ["Public_Event_Crowding", "Transit_Delay", "Support_Team", "Service_Restored"],
            ["Public_Event_Crowding", "Transit_Delay", "Public_Notice", "Support_Team"],
            ["Heat_Advisory", "Public_Notice", "Support_Team", "Service_Restored"],
            ["Heat_Advisory", "Service_Disruption", "Support_Team", "Service_Restored"],
            ["Waste_Collection_Delay", "Public_Notice", "Cleanup", "Service_Restored"],
            ["Waste_Collection_Delay", "Service_Disruption", "Support_Team", "Service_Restored"],
            ["Street_Light_Issue", "Service_Disruption", "Repair_Team", "Repair_Complete"],
        ]
        sequences.extend(canonical_20)

    weights = [tpl["weight"] for tpl in INCIDENT_TEMPLATES]
    while len(sequences) < count:
        tpl = rng.choices(INCIDENT_TEMPLATES, weights=weights, k=1)[0]
        stages = tpl["stages"]
        seq = [rng.choice(stage) for stage in stages]
        # Occasional sequence length variations (3 to 5 events)
        if rng.random() < 0.15 and len(seq) > 3:
            # Short incident
            seq = seq[:3]
        sequences.append(seq)

    # Return exactly count items
    return validate_incidents(sequences[:count])
