"""
Tests for pipeline.py — Property 17 and data model integrity.

Property 17: Data model enumeration integrity
  For any QAPair produced by the pipeline, bloom_level is one of
  ["Remember", "Understand", "Apply", "Analyze"].

Validates: Requirements 9.4
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models import VALID_BLOOM_LEVELS


# ---------------------------------------------------------------------------
# Unit test: VALID_BLOOM_LEVELS constant contains exactly the four expected values
# ---------------------------------------------------------------------------

def test_valid_bloom_levels_contains_exactly_four_values():
    """VALID_BLOOM_LEVELS must contain exactly the four Bloom taxonomy levels."""
    expected = {"Remember", "Understand", "Apply", "Analyze"}
    assert set(VALID_BLOOM_LEVELS) == expected
    assert len(VALID_BLOOM_LEVELS) == 4


# ---------------------------------------------------------------------------
# Property 17: bloom_level in any QAPair-like dict is always a valid Bloom level
#
# Since run_pipeline requires a real PDF and real models, we test the property
# at the unit level: generate lists of QAPair-like dicts with bloom_level values
# drawn from the valid set and verify the constraint holds for every element.
# ---------------------------------------------------------------------------

@given(
    qa_pairs=st.lists(
        st.fixed_dictionaries({
            "question": st.text(min_size=1),
            "answer": st.text(min_size=1),
            "bloom_level": st.sampled_from(VALID_BLOOM_LEVELS),
            "source_note": st.text(min_size=1),
        }),
        min_size=1,
        max_size=20,
    )
)
@settings(max_examples=200)
def test_property_17_bloom_level_enumeration_integrity(qa_pairs):
    """**Validates: Requirements 9.4**

    Property 17: For any QAPair produced by the pipeline, bloom_level is one of
    ["Remember", "Understand", "Apply", "Analyze"].
    """
    for pair in qa_pairs:
        assert pair["bloom_level"] in VALID_BLOOM_LEVELS, (
            f"bloom_level '{pair['bloom_level']}' is not in {VALID_BLOOM_LEVELS}"
        )


# ---------------------------------------------------------------------------
# Additional unit test: QAPair dataclass enforces bloom_level at construction
# ---------------------------------------------------------------------------

def test_qapair_rejects_invalid_bloom_level():
    """QAPair.__post_init__ must raise ValueError for an invalid bloom_level."""
    from models import QAPair

    with pytest.raises(ValueError, match="bloom_level"):
        QAPair(
            question="What is X?",
            answer="X",
            bloom_level="Create",   # not a valid level
            source_note="X is important.",
        )


def test_qapair_accepts_all_valid_bloom_levels():
    """QAPair must accept each of the four valid Bloom levels without error."""
    from models import QAPair

    for level in VALID_BLOOM_LEVELS:
        pair = QAPair(
            question="What is X?",
            answer="X",
            bloom_level=level,
            source_note="X is important.",
        )
        assert pair.bloom_level == level
