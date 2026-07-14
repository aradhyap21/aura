"""
RemediationService: Generates adaptive recovery learning paths
based on weak topics, misconceptions, and Bloom level.
"""

from adaptive_engine import bloom_level_name


REMEDIATION_STEPS = {
    "critical": [
        "review_note",
        "definition_drill",
        "example_problem",
        "active_recall",
        "bloom_l1_question",
        "bloom_l2_question",
    ],
    "weak": [
        "review_note",
        "example_problem",
        "active_recall",
        "bloom_l2_question",
        "bloom_l3_question",
    ],
    "moderate": [
        "example_problem",
        "active_recall",
        "bloom_l3_question",
        "bloom_l4_question",
    ],
}

STEP_LABELS = {
    "review_note":        ("📖", "Review Notes", "Re-read the structured notes for this topic"),
    "definition_drill":   ("📝", "Definition Drill", "Memorize key definitions and terms"),
    "example_problem":    ("🔢", "Worked Example", "Study a solved example step by step"),
    "active_recall":      ("🧠", "Active Recall", "Answer a free-text recall question"),
    "bloom_l1_question":  ("💡", "Remember Question", "Answer a basic recall question"),
    "bloom_l2_question":  ("🔍", "Understand Question", "Explain the concept in your own words"),
    "bloom_l3_question":  ("⚙️", "Apply Question", "Solve an application problem"),
    "bloom_l4_question":  ("🔬", "Analyze Question", "Compare, contrast, or break down the concept"),
}


def generate_remediation_path(
    topic: str,
    mastery_score: float,
    misconception_score: float,
    bloom_level: int,
) -> dict:
    """
    Generate a personalized recovery path for a weak topic.

    Returns:
        {topic, severity, steps: [{icon, title, description, step_id}]}
    """
    if mastery_score < 0.4 or misconception_score > 0.6:
        severity = "critical"
    elif mastery_score < 0.65:
        severity = "weak"
    else:
        severity = "moderate"

    step_ids = REMEDIATION_STEPS[severity]
    steps = []
    for step_id in step_ids:
        icon, title, desc = STEP_LABELS.get(step_id, ("▶️", step_id, ""))
        steps.append({"step_id": step_id, "icon": icon, "title": title, "description": desc})

    return {
        "topic": topic,
        "severity": severity,
        "mastery_score": mastery_score,
        "misconception_score": misconception_score,
        "current_bloom_level": bloom_level,
        "target_bloom_level": min(bloom_level + 1, 5),
        "steps": steps,
    }
