"""
AdaptiveEngine: Bloom-level difficulty progression and confidence-weighted scoring.

Bloom levels:
  1 = Remember
  2 = Understand
  3 = Apply
  4 = Analyze
  5 = Evaluate / Create
"""

BLOOM_LEVELS = {
    1: "Remember",
    2: "Understand",
    3: "Apply",
    4: "Analyze",
    5: "Evaluate",
}

BLOOM_NAMES_INV = {v: k for k, v in BLOOM_LEVELS.items()}


def next_bloom_level(score: float, current_level: int) -> int:
    """
    Adjust Bloom level based on accuracy score.
    score >= 0.8  → increase by 1
    score < 0.5   → decrease by 1
    else          → stay same
    """
    if score >= 0.8:
        return min(current_level + 1, 5)
    elif score < 0.5:
        return max(current_level - 1, 1)
    return current_level


def bloom_level_name(level: int) -> str:
    return BLOOM_LEVELS.get(level, "Remember")


def confidence_weighted_score(is_correct: bool, confidence: int) -> float:
    """
    S = 0.7 * C + 0.3 * (K / 5)
    C = correctness (0 or 1)
    K = confidence (1–5)
    """
    C = 1.0 if is_correct else 0.0
    K = max(1, min(5, confidence))
    return round(0.7 * C + 0.3 * (K / 5), 4)


def is_misconception(is_correct: bool, confidence: int) -> bool:
    """Flag as misconception if wrong answer with high confidence (>= 4)."""
    return (not is_correct) and (confidence >= 4)


def difficulty_from_bloom(level: int) -> str:
    """Map Bloom level to decay difficulty for retention model."""
    if level <= 2:
        return "easy"
    elif level <= 3:
        return "medium"
    return "hard"
