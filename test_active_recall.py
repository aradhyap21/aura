"""
Property-based tests for active_recall.py

Properties covered:
  - Property 14: evaluate_answer score is always in range (Validates: Requirements 7.3, 7.5)
  - Property 15: Identical answers score 1.0 (Validates: Requirements 7.4)
  - Property 16: RecallPrompt structural completeness (Validates: Requirements 7.1, 7.7, 9.5)
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from active_recall import evaluate_answer, build_recall_prompts

VALID_PROMPT_TYPES = {"qa", "cloze"}

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

any_string = st.text()

non_empty_text = st.text(min_size=1).filter(lambda s: s.strip())

@st.composite
def qa_pairs_strategy(draw):
    """Generate a list of qa_pair dicts with non-empty question and answer."""
    n = draw(st.integers(min_value=0, max_value=10))
    pairs = []
    for _ in range(n):
        question = draw(non_empty_text)
        answer = draw(non_empty_text)
        bloom_level = draw(st.sampled_from(["Remember", "Understand", "Apply", "Analyze"]))
        source_note = draw(non_empty_text)
        pairs.append({
            "question": question,
            "answer": answer,
            "bloom_level": bloom_level,
            "source_note": source_note,
        })
    return pairs


@st.composite
def notes_strategy(draw):
    """Generate a list of note strings (may be empty list)."""
    return draw(st.lists(non_empty_text, min_size=0, max_size=10))


# ---------------------------------------------------------------------------
# Property 14: evaluate_answer score is always in range
# ---------------------------------------------------------------------------

@given(any_string, any_string)
@settings(max_examples=500)
def test_property14_score_always_in_range(user_answer, expected_answer):
    """
    Property 14: evaluate_answer score is always in range.

    For any two strings (including empty), evaluate_answer returns a score in
    [0.0, 1.0], and is_correct is True if and only if score >= 0.5.

    **Validates: Requirements 7.3, 7.5**
    """
    result = evaluate_answer(user_answer, expected_answer)

    assert "score" in result, "Result must contain 'score' key"
    assert "is_correct" in result, "Result must contain 'is_correct' key"
    assert "expected" in result, "Result must contain 'expected' key"

    score = result["score"]
    assert 0.0 <= score <= 1.0, f"Score must be in [0.0, 1.0], got {score}"
    assert result["is_correct"] == (score >= 0.5), (
        f"is_correct must be True iff score >= 0.5: score={score}, "
        f"is_correct={result['is_correct']}"
    )


# ---------------------------------------------------------------------------
# Property 15: Identical answers score 1.0
# ---------------------------------------------------------------------------

@given(any_string)
@settings(max_examples=500)
def test_property15_identical_answers_score_1(s):
    """
    Property 15: Identical answers score 1.0.

    For any string s, evaluate_answer(s, s) returns score = 1.0.

    **Validates: Requirements 7.4**
    """
    result = evaluate_answer(s, s)
    assert result["score"] == 1.0, (
        f"evaluate_answer(s, s) must return score=1.0 for s={s!r}, "
        f"got {result['score']}"
    )


@given(non_empty_text)
@settings(max_examples=500)
def test_property15_case_variant_scores_1(s):
    """
    Property 15 (case variant): For any string s and its case-variant s'
    (same tokens, different casing), evaluate_answer(s, s') returns score = 1.0.

    We construct s' by lowercasing s so that token sets are identical when
    compared case-insensitively (both sides lower to the same tokens).

    **Validates: Requirements 7.4**
    """
    s_lower = s.lower()
    result = evaluate_answer(s, s_lower)
    assert result["score"] == 1.0, (
        f"evaluate_answer(s, s.lower()) must return score=1.0 for s={s!r}, "
        f"got {result['score']}"
    )


# ---------------------------------------------------------------------------
# Property 16: RecallPrompt structural completeness
# ---------------------------------------------------------------------------

@given(notes_strategy(), qa_pairs_strategy())
@settings(max_examples=300)
def test_property16_recall_prompt_structural_completeness(notes, qa_pairs):
    """
    Property 16: RecallPrompt structural completeness.

    For any RecallPrompt produced by build_recall_prompts:
    - prompt, expected, and prompt_type fields are all present and non-empty
    - prompt_type is either "qa" or "cloze"

    **Validates: Requirements 7.1, 7.7, 9.5**
    """
    prompts = build_recall_prompts(notes, qa_pairs)

    for i, p in enumerate(prompts):
        assert "prompt" in p, f"Prompt {i} missing 'prompt' field"
        assert "expected" in p, f"Prompt {i} missing 'expected' field"
        assert "prompt_type" in p, f"Prompt {i} missing 'prompt_type' field"

        assert p["prompt"], f"Prompt {i} 'prompt' must be non-empty"
        assert p["expected"], f"Prompt {i} 'expected' must be non-empty"
        assert p["prompt_type"], f"Prompt {i} 'prompt_type' must be non-empty"

        assert p["prompt_type"] in VALID_PROMPT_TYPES, (
            f"Prompt {i} 'prompt_type' must be 'qa' or 'cloze', "
            f"got {p['prompt_type']!r}"
        )
