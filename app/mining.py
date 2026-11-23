"""Frequent Event Pattern Mining for IncidentIQ.

Integrates with the `thilimarketbasket` package (specifically thilimarketbasket.mine)
to discover frequent event co-occurrences and association rules.
Provides a transparent fallback if the package is unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import Counter
from itertools import combinations
import math

from .dataset import load_incidents, validate_incidents

logger = logging.getLogger(__name__)

# Check thilimarketbasket availability
THILI_AVAILABLE = False
try:
    from thilimarketbasket.mine import frequent_itemsets as thili_frequent_itemsets
    from thilimarketbasket.mine import rules as thili_rules
    THILI_AVAILABLE = True
except ImportError:
    logger.warning(
        "Package 'thilimarketbasket' is not installed or could not be imported. "
        "A transparent fallback algorithm will be used."
    )


def _local_apriori(baskets: List[List[str]], min_support: float, max_size: int = 5) -> Dict[frozenset, float]:
    """Pure-Python Apriori counting fallback matching thilimarketbasket.mine.frequent_itemsets."""
    n = len(baskets)
    min_count = max(1, math.ceil(min_support * n - 1e-9))
    item_counts = Counter()
    for basket in baskets:
        for item in set(basket):
            item_counts[item] += 1

    level = {frozenset([item]): count / n for item, count in item_counts.items() if count >= min_count}
    found = dict(level)
    size = 1

    while level and size < max_size:
        prev_itemsets = list(level.keys())
        candidates = set()
        for i in range(len(prev_itemsets)):
            for j in range(i + 1, len(prev_itemsets)):
                union = prev_itemsets[i] | prev_itemsets[j]
                if len(union) == size + 1:
                    candidates.add(union)

        next_level = {}
        for candidate in candidates:
            # Check all subsets are frequent
            subsets = combinations(candidate, size)
            if all(frozenset(s) in level for s in subsets):
                count = sum(1 for basket in baskets if candidate.issubset(set(basket)))
                if count >= min_count:
                    next_level[candidate] = count / n

        level = next_level
        found.update(level)
        size += 1

    return found


def _local_rules(itemsets: Dict[frozenset, float], min_confidence: float = 0.3) -> List[Tuple[frozenset, frozenset, float, float, float]]:
    """Generate association rules matching thilimarketbasket.mine.rules."""
    found = []
    for items, support in itemsets.items():
        if len(items) < 2:
            continue
        for k in range(1, len(items)):
            for antecedent in map(frozenset, combinations(sorted(items), k)):
                consequent = items - antecedent
                ant_sup = itemsets.get(antecedent, 0.0)
                if ant_sup <= 0:
                    continue
                confidence = support / ant_sup
                con_sup = itemsets.get(consequent, 0.0)
                lift = (confidence / con_sup) if con_sup > 0 else 0.0
                if confidence >= min_confidence:
                    found.append((antecedent, consequent, support, confidence, lift))
    return sorted(found, key=lambda rule: (-rule[4], -rule[2]))


def mine_frequent_patterns(
    reports: Optional[Sequence[Sequence[str]]] = None,
    min_support: float = 0.20,
    max_size: int = 4,
    min_confidence: float = 0.30,
    force_fallback: bool = False,
) -> Dict[str, Any]:
    """Mine frequent event itemsets and association rules using thilimarketbasket.

    Args:
        reports: Sequence of incident event sequences. Defaults to loading from data.
        min_support: Minimum support threshold in [0.0, 1.0].
        max_size: Maximum pattern size to search.
        min_confidence: Minimum confidence for association rules.
        force_fallback: If True, forces the local fallback even if package is installed (useful for testing).

    Returns:
        Dict containing patterns, rules, metadata, and transparent engine source.
    """
    if reports is None:
        sequences = load_incidents()
    else:
        sequences = validate_incidents(reports)

    use_thili = THILI_AVAILABLE and not force_fallback
    engine_name = "thilimarketbasket" if use_thili else "fallback_apriori"

    if use_thili:
        itemsets_dict = thili_frequent_itemsets(sequences, min_support=min_support, max_size=max_size)
        raw_rules = thili_rules(itemsets_dict, min_confidence=min_confidence)
    else:
        itemsets_dict = _local_apriori(sequences, min_support=min_support, max_size=max_size)
        raw_rules = _local_rules(itemsets_dict, min_confidence=min_confidence)

    n_reports = len(sequences)
    patterns: List[Dict[str, Any]] = []
    for itemset, support in itemsets_dict.items():
        count = int(round(support * n_reports))
        patterns.append({
            "pattern": sorted(list(itemset)),
            "support": round(float(support), 4),
            "count": count,
            "size": len(itemset),
        })

    # Sort patterns: longer size first, then higher support, then alphabetical
    patterns.sort(key=lambda r: (-r["size"], -r["support"], r["pattern"]))

    formatted_rules: List[Dict[str, Any]] = []
    for ant, con, sup, conf, lift in raw_rules:
        formatted_rules.append({
            "antecedent": sorted(list(ant)),
            "consequent": sorted(list(con)),
            "support": round(float(sup), 4),
            "confidence": round(float(conf), 4),
            "lift": round(float(lift), 4),
        })

    return {
        "engine": engine_name,
        "is_fallback": not use_thili,
        "total_sequences": n_reports,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "patterns": patterns,
        "rules": formatted_rules,
    }


def milestone_2(
    reports: Optional[Sequence[Sequence[str]]] = None,
    min_support: float = 0.20,
    max_pattern_length: int = 3,
) -> List[Dict[str, Any]]:
    """Milestone 2 interface: return frequent patterns list with support counts."""
    result = mine_frequent_patterns(
        reports=reports,
        min_support=min_support,
        max_size=max_pattern_length,
    )
    return result["patterns"]
