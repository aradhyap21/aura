"""
Property-based tests for mcq_builder.py

Properties covered:
  - Property 11: MCQ output length equals input length (Validates: Requirements 5.1)
  - Property 12: MCQItem structural integrity (Validates: Requirements 5.2, 5.4, 9.2, 9.3)
"""

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st

from mcq_builder import build_mcqs

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze"]

# ---------------------------------------------------------------------------
# Strategy: generate a list of qa_pair dicts with min_size=4 and unique answers
# ---------------------------------------------------------------------------

@st.composite
def qa_pairs_strategy(draw):
    """Generate a list of qa_pair dicts with unique answer strings, min_size=4."""
    n = draw(st.integers(min_value=4, max_value=20))
    # Draw n unique answer strings
    answers = draw(
        st.lists(
            st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    bloom_levels = BLOOM_LEVELS
    pairs = []
    for i, answer in enumerate(answers):
        question = draw(st.text(min_size=1, max_size=100).filter(lambda s: s.strip()))
        bloom_level = bloom_levels[i % len(bloom_levels)]
        source_note = draw(st.text(min_size=1, max_size=200).filter(lambda s: s.strip()))
        pairs.append({
            "question": question,
            "answer": answer,
            "bloom_level": bloom_level,
            "source_note": source_note,
        })
    return pairs


# ---------------------------------------------------------------------------
# Property 11: MCQ output length equals input length
# ---------------------------------------------------------------------------

@given(qa_pairs_strategy())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_property11_mcq_output_length_equals_input_length(qa_pairs):
    """
    Property 11: MCQ output length equals input length.

    For any list of QAPairs with length >= 4, build_mcqs returns a list of
    MCQItems of the same length.

    **Validates: Requirements 5.1**
    """
    assert len(qa_pairs) >= 4, "Precondition: qa_pairs must have at least 4 items"
    mcqs = build_mcqs(qa_pairs)
    assert len(mcqs) == len(qa_pairs), (
        f"build_mcqs must return one MCQItem per QAPair: "
        f"expected {len(qa_pairs)}, got {len(mcqs)}"
    )


# ---------------------------------------------------------------------------
# Property 12: MCQItem structural integrity
# ---------------------------------------------------------------------------

@given(qa_pairs_strategy())
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_property12_mcqitem_structural_integrity(qa_pairs):
    """
    Property 12: MCQItem structural integrity.

    For any MCQItem produced by build_mcqs:
    - options list contains exactly 4 elements
    - correct_index is in the range [0, 3]
    - options[correct_index] equals the original answer from the source QAPair

    **Validates: Requirements 5.2, 5.4, 9.2, 9.3**
    """
    assert len(qa_pairs) >= 4, "Precondition: qa_pairs must have at least 4 items"
    mcqs = build_mcqs(qa_pairs)

    for i, (mcq, source) in enumerate(zip(mcqs, qa_pairs)):
        assert len(mcq["options"]) == 4, (
            f"MCQItem {i}: options must have exactly 4 elements, "
            f"got {len(mcq['options'])}"
        )
        assert mcq["correct_index"] in range(4), (
            f"MCQItem {i}: correct_index must be in [0, 3], "
            f"got {mcq['correct_index']}"
        )
        assert mcq["options"][mcq["correct_index"]] == source["answer"], (
            f"MCQItem {i}: options[correct_index] must equal the original answer. "
            f"Expected {source['answer']!r}, "
            f"got {mcq['options'][mcq['correct_index']]!r}"
        )
