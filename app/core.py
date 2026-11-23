"""Core gameplay and ML-lite components for IncidentIQ.

All scenarios are fictional, low-risk civic-service training data.
"""
from collections import Counter, defaultdict
from itertools import combinations
import random

# Fictional civic-service incident sequences, represented as ordered event lists.
INCIDENT_SEQUENCES = [
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


# Resources are abstract training points; requirements are invented for game scoring.
RESOURCE_TYPES = ("teams", "equipment", "communications")
RESOURCE_COSTS = {"teams": 10, "equipment": 5, "communications": 5}
RESOURCE_NEEDS = {
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


# Milestone 1

def milestone_1(reports=None):
    """Validate and return ordered reports (sequences of event strings)."""
    reports = INCIDENT_SEQUENCES if reports is None else reports
    validated = []
    for report in reports:
        if not isinstance(report, (list, tuple)) or not report or not all(isinstance(e, str) and e.strip() for e in report):
            raise ValueError("Each incident report must be a non-empty sequence of event names.")
        validated.append(list(report))
    return validated


# Milestone 2: sequence-pattern mining with support counting. The named package's API
# is not assumed; this deterministic implementation provides the same pattern output.

def milestone_2(reports=None, min_support=0.20, max_pattern_length=3):
    """Find frequent ordered subsequences and return support counts/percentages."""
    sequences = milestone_1(reports)
    counts = Counter()
    for seq in sequences:
        seen = set()
        for length in range(1, min(max_pattern_length, len(seq)) + 1):
            for positions in combinations(range(len(seq)), length):
                pattern = tuple(seq[i] for i in positions)
                seen.add(pattern)
        counts.update(seen)
    threshold = max(1, int(min_support * len(sequences) + 0.999999))
    result = [
        {"pattern": list(pattern), "count": count, "support": count / len(sequences)}
        for pattern, count in counts.items() if count >= threshold
    ]
    result.sort(key=lambda row: (-len(row["pattern"]), -row["support"], row["pattern"]))
    return result


# Milestone 3: lightweight sequential next-event model (n-gram counts).
class NextEventModel:
    def __init__(self, reports=None):
        self.reports = milestone_1(reports)
        self.transitions = defaultdict(Counter)
        self.unigram = Counter()
        self.trained = False

    def train(self):
        for sequence in self.reports:
            for i, event in enumerate(sequence):
                self.unigram[event] += 1
                for n in (1, 2, 3):
                    context = tuple(sequence[max(0, i - n):i])
                    if context:
                        self.transitions[context][event] += 1
        self.trained = True
        return self

    def predict(self, context, top_k=3):
        if not self.trained:
            self.train()
        context = tuple(context or [])
        for n in range(min(3, len(context)), 0, -1):
            key = context[-n:]
            if key in self.transitions and self.transitions[key]:
                counts = self.transitions[key]
                total = sum(counts.values())
                return [(event, count / total) for event, count in counts.most_common(top_k)]
        total = sum(self.unigram.values()) or 1
        return [(event, count / total) for event, count in self.unigram.most_common(top_k)]


def milestone_3(reports=None):
    """Train a next-event model from the sequence corpus; no external weights needed."""
    return NextEventModel(reports).train()


# Milestone 4: player round and resource allocation

def resource_cost(allocation):
    return sum(max(0, int(allocation.get(resource, 0))) * RESOURCE_COSTS[resource] for resource in RESOURCE_TYPES)


def milestone_4(rounds=5, reports=None, seed=None, budget=100):
    """Play a CLI game; optionally pass reports/seed for reproducible testing."""
    sequences = milestone_1(reports)
    model = milestone_3(sequences)
    rng = random.Random(seed)
    points_left = budget
    round_results = []
    print("\n=== INCIDENTIQ: Predict & Respond ===")
    print("Fictional civic-service scenarios. Resource points are game abstractions.\n")
    for round_no in range(1, rounds + 1):
        sequence = rng.choice(sequences)
        idx = rng.randrange(1, len(sequence))
        context = sequence[:idx]
        actual = sequence[idx]
        predictions = model.predict(context)
        print(f"\n--- Round {round_no}/{rounds} | Budget remaining: {points_left} ---")
        print("Situation: " + " -> ".join(context))
        print("Model predictions:")
        for i, (event, prob) in enumerate(predictions, 1):
            print(f"  {i}. {event} ({prob:.0%})")
        print("Choose prediction (number, or enter event name):")
        choice = input("Prediction: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(predictions):
            player_prediction = predictions[int(choice) - 1][0]
        else:
            player_prediction = choice
        allocation = {}
        for resource in RESOURCE_TYPES:
            while True:
                raw = input(f"Allocate {resource} (whole number): ").strip()
                try:
                    value = int(raw)
                    if value < 0:
                        raise ValueError
                    allocation[resource] = value
                    break
                except ValueError:
                    print("Enter a non-negative whole number.")
        spent = resource_cost(allocation)
        while spent > points_left:
            print(f"Allocation costs {spent}, but only {points_left} budget points remain. Re-enter allocation.")
            allocation = {}
            for resource in RESOURCE_TYPES:
                while True:
                    try:
                        value = int(input(f"Allocate {resource} (whole number): ").strip())
                        if value < 0: raise ValueError
                        allocation[resource] = value
                        break
                    except ValueError:
                        print("Enter a non-negative whole number.")
            spent = resource_cost(allocation)
        points_left -= spent
        correct = player_prediction == actual
        needs = RESOURCE_NEEDS.get(actual, {r: 1 for r in RESOURCE_TYPES})
        coverage = sum(min(allocation.get(r, 0), needs.get(r, 0)) for r in RESOURCE_TYPES) / max(1, sum(needs.values()))
        over_allocated = sum(max(0, allocation.get(r, 0) - needs.get(r, 0)) for r in RESOURCE_TYPES)
        efficiency = max(0.0, min(1.0, coverage - 0.05 * over_allocated))
        print(f"Actual next event: {actual}")
        print("Your prediction: " + ("CORRECT" if correct else "incorrect"))
        print(f"Resource coverage: {coverage:.0%} | Resource efficiency: {efficiency:.0%} | Spent: {spent}")
        round_results.append({"prediction": player_prediction, "actual": actual, "correct": correct,
                           "allocation": allocation, "spent": spent, "coverage": coverage,
                           "efficiency": efficiency, "model_prediction": predictions[0][0] if predictions else None})
    return round_results


# Milestone 5: evaluation metrics

def milestone_5(results):
    """Return prediction accuracy, mean resource efficiency, and weighted score."""
    if not results:
        return {"prediction_accuracy": 0.0, "resource_efficiency": 0.0, "overall_score": 0.0, "rounds": 0}
    accuracy = sum(bool(r.get("correct")) for r in results) / len(results)
    efficiency = sum(float(r.get("efficiency", 0)) for r in results) / len(results)
    return {"prediction_accuracy": round(accuracy * 100, 2),
            "resource_efficiency": round(efficiency * 100, 2),
            "overall_score": round((0.6 * accuracy + 0.4 * efficiency) * 100, 2),
            "rounds": len(results)}
