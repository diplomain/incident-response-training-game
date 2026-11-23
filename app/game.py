"""Interactive and Simulated Game Engine for IncidentIQ (Milestone 5).

Manages rounds, model predictions, player allocation, budget constraints,
event reveals, scoring, and session persistence.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .dataset import load_incidents, validate_incidents
from .prediction import NGramPredictor, milestone_3
from .neural_lm import NeuralNextEventLM, DEFAULT_WEIGHTS_PATH
from .storage import Store

# Fictional civic training resources and scoring tables
RESOURCE_TYPES = ("teams", "equipment", "communications")
RESOURCE_COSTS = {"teams": 10, "equipment": 5, "communications": 5}

RESOURCE_NEEDS: Dict[str, Dict[str, int]] = {
    "Waterlogging": {"teams": 2, "equipment": 3, "communications": 1},
    "Road_Blockage": {"teams": 2, "equipment": 2, "communications": 1},
    "Traffic_Disruption": {"teams": 1, "equipment": 1, "communications": 2},
    "Service_Disruption": {"teams": 1, "equipment": 2, "communications": 2},
    "Public_Notice": {"teams": 0, "equipment": 0, "communications": 2},
    "Cleanup": {"teams": 2, "equipment": 2, "communications": 1},
    "Drainage_Check": {"teams": 1, "equipment": 2, "communications": 1},
    "Repair_Team": {"teams": 2, "equipment": 2, "communications": 1},
    "Repair_Complete": {"teams": 1, "equipment": 1, "communications": 1},
    "Transit_Delay": {"teams": 1, "equipment": 1, "communications": 2},
    "Support_Team": {"teams": 2, "equipment": 1, "communications": 1},
    "Service_Restored": {"teams": 1, "equipment": 1, "communications": 1},
    "Heat_Advisory": {"teams": 1, "equipment": 1, "communications": 2},
    "Waste_Collection_Delay": {"teams": 1, "equipment": 2, "communications": 1},
    "Street_Light_Issue": {"teams": 1, "equipment": 2, "communications": 1},
    "Power_Outage": {"teams": 1, "equipment": 2, "communications": 2},
    "Water_Supply_Issue": {"teams": 1, "equipment": 2, "communications": 2},
    "Heavy_Rain": {"teams": 1, "equipment": 1, "communications": 1},
    "Public_Event_Crowding": {"teams": 2, "equipment": 1, "communications": 2},
}


def resource_cost(allocation: Dict[str, int]) -> int:
    """Calculate budget cost for an allocation dictionary."""
    return sum(
        max(0, int(allocation.get(res, 0))) * RESOURCE_COSTS[res]
        for res in RESOURCE_TYPES
    )


def calculate_round_metrics(
    actual_event: str,
    allocation: Dict[str, int],
) -> Tuple[float, float]:
    """Calculate resource coverage and net resource efficiency for an event."""
    needs = RESOURCE_NEEDS.get(actual_event, {r: 1 for r in RESOURCE_TYPES})
    total_needed = max(1, sum(needs.values()))
    covered = sum(min(max(0, allocation.get(r, 0)), needs.get(r, 0)) for r in RESOURCE_TYPES)
    coverage = covered / total_needed

    over_allocated = sum(max(0, allocation.get(r, 0) - needs.get(r, 0)) for r in RESOURCE_TYPES)
    efficiency = max(0.0, min(1.0, coverage - 0.05 * over_allocated))
    return coverage, efficiency


def milestone_5(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate session scores: prediction accuracy, mean efficiency, and overall weighted score."""
    if not results:
        return {
            "prediction_accuracy": 0.0,
            "resource_efficiency": 0.0,
            "overall_score": 0.0,
            "rounds": 0,
            "total_spent": 0,
            "remaining_budget": 0,
        }

    rounds = len(results)
    accuracy = sum(1 for r in results if r.get("correct")) / rounds
    efficiency = sum(float(r.get("efficiency", 0.0)) for r in results) / rounds
    overall = 0.6 * accuracy + 0.4 * efficiency
    total_spent = sum(int(r.get("spent", 0)) for r in results)

    return {
        "prediction_accuracy": round(accuracy * 100, 2),
        "resource_efficiency": round(efficiency * 100, 2),
        "overall_score": round(overall * 100, 2),
        "rounds": rounds,
        "total_spent": total_spent,
    }


def play_game(
    rounds: int = 5,
    budget: int = 100,
    reports: Optional[Sequence[Sequence[str]]] = None,
    predictor: Any = None,
    predictor_type: str = "ngram",
    interactive: bool = True,
    simulated_strategy: str = "model",
    seed: Optional[int] = None,
    store_session: bool = True,
) -> Dict[str, Any]:
    """Play a game session of IncidentIQ (interactive in terminal or simulated).

    Args:
        rounds: Number of incident scenarios to play.
        budget: Total starting budget for allocating resources.
        reports: Optional sequence dataset. Defaults to data/incident_sequences.json.
        predictor: Model instance. If None, loaded based on predictor_type.
        predictor_type: "ngram" or "neural".
        interactive: If True, prompts user input via terminal. If False, runs simulation.
        simulated_strategy: "model" (follows top model prediction) or "random".
        seed: Random seed for scenario selection.
        store_session: If True, persists session score in Store (data/app.json).

    Returns:
        Dict with round details and final session scores.
    """
    sequences = load_incidents() if reports is None else validate_incidents(reports)

    # Initialize predictor if needed
    if predictor is None:
        if predictor_type == "neural" and DEFAULT_WEIGHTS_PATH.exists():
            neural = NeuralNextEventLM()
            neural.load_weights(DEFAULT_WEIGHTS_PATH)
            predictor = neural
        else:
            predictor = milestone_3(sequences)

    rng = random.Random(seed)
    remaining_budget = budget
    round_records: List[Dict[str, Any]] = []

    if interactive:
        print("\n=======================================================")
        print("          INCIDENTIQ: PREDICT & RESPOND               ")
        print("=======================================================")
        print("Educational civic-service simulation. Game tokens are abstract.")
        print(f"Starting Budget: {budget} | Rounds: {rounds} | Model: {getattr(predictor, 'name', 'Next-Event Model')}\n")

    for round_no in range(1, rounds + 1):
        seq = rng.choice(sequences)
        split_idx = rng.randint(1, len(seq) - 1)
        context = seq[:split_idx]
        actual_event = seq[split_idx]

        predictions = predictor.predict(context, top_k=3)

        if interactive:
            print(f"\n--- Round {round_no}/{rounds} | Budget: {remaining_budget} pts ---")
            print("Situation Context: " + " -> ".join(context))
            print("AI Suggestions:")
            for i, (ev, prob) in enumerate(predictions, 1):
                print(f"  [{i}] {ev} ({prob:.0%})")

        # Select prediction
        if interactive:
            while True:
                choice = input("Your prediction (1-3 or event name): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(predictions):
                    player_pred = predictions[int(choice) - 1][0]
                    break
                elif choice:
                    player_pred = choice
                    break
                print("Please enter a valid choice.")
        else:
            if simulated_strategy == "model" and predictions:
                player_pred = predictions[0][0]
            else:
                player_pred = rng.choice(list(RESOURCE_NEEDS.keys()))

        # Resource allocation
        allocation: Dict[str, int] = {}
        if interactive:
            while True:
                print("Allocate Response Resources (teams: 10pts, equipment: 5pts, comms: 5pts):")
                temp_alloc = {}
                for r in RESOURCE_TYPES:
                    while True:
                        raw = input(f"  Allocate {r}: ").strip()
                        try:
                            val = int(raw)
                            if val < 0:
                                raise ValueError
                            temp_alloc[r] = val
                            break
                        except ValueError:
                            print("    Must be a non-negative integer.")
                cost = resource_cost(temp_alloc)
                if cost > remaining_budget:
                    print(f"  Allocation costs {cost} pts, but only {remaining_budget} pts remaining! Try again.")
                else:
                    allocation = temp_alloc
                    spent = cost
                    break
        else:
            # Automated simulation allocation: allocate according to predicted needs if affordable
            target_needs = RESOURCE_NEEDS.get(player_pred, {"teams": 1, "equipment": 1, "communications": 1})
            temp_alloc = dict(target_needs)
            cost = resource_cost(temp_alloc)
            while cost > remaining_budget and any(temp_alloc[r] > 0 for r in RESOURCE_TYPES):
                # Scale down allocation to fit budget
                for r in RESOURCE_TYPES:
                    if temp_alloc[r] > 0:
                        temp_alloc[r] -= 1
                        break
                cost = resource_cost(temp_alloc)
            if cost > remaining_budget:
                temp_alloc = {r: 0 for r in RESOURCE_TYPES}
                cost = 0
            allocation = temp_alloc
            spent = cost

        remaining_budget -= spent
        correct = (player_pred == actual_event)
        coverage, efficiency = calculate_round_metrics(actual_event, allocation)

        if interactive:
            print(f"\n>> Actual Next Event: {actual_event}")
            print(f">> Prediction Result: {'CORRECT (+)' if correct else 'INCORRECT (-)'}")
            print(f">> Coverage: {coverage:.0%} | Resource Efficiency: {efficiency:.0%} | Cost: {spent} pts")

        round_records.append({
            "round": round_no,
            "context": context,
            "actual": actual_event,
            "prediction": player_pred,
            "correct": correct,
            "allocation": allocation,
            "spent": spent,
            "coverage": coverage,
            "efficiency": efficiency,
            "model_suggestions": predictions,
            "remaining_budget": remaining_budget,
        })

    session_scores = milestone_5(round_records)
    session_scores["remaining_budget"] = remaining_budget

    if interactive:
        print("\n=======================================================")
        print("                   FINAL PERFORMANCE                   ")
        print("=======================================================")
        print(f"Prediction Accuracy : {session_scores['prediction_accuracy']:.1f}%")
        print(f"Resource Efficiency : {session_scores['resource_efficiency']:.1f}%")
        print(f"Overall Game Score  : {session_scores['overall_score']:.1f} / 100")
        print(f"Remaining Budget    : {remaining_budget} pts")
        print("=======================================================\n")

    if store_session:
        try:
            store = Store()
            history = store.get("game_history", [])
            history.append({
                "rounds": rounds,
                "scores": session_scores,
                "model": getattr(predictor, "name", "Model"),
            })
            store.put("game_history", history)
        except Exception:
            pass

    return {
        "round_records": round_records,
        "session_scores": session_scores,
    }


def milestone_4(rounds: int = 5, reports: Optional[Sequence[Sequence[str]]] = None, seed: Optional[int] = None, budget: int = 100) -> List[Dict[str, Any]]:
    """Milestone 4 interface: run gameplay and return round records."""
    res = play_game(rounds=rounds, budget=budget, reports=reports, seed=seed, interactive=True)
    return res["round_records"]
