"""
Property tests for notes_generator.generate_notes.

Property 6: Notes filtering excludes short sentences
Validates: Requirements 3.1, 3.2
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from notes_generator import generate_notes


@given(st.text())
@settings(max_examples=200)
def test_property6_notes_filtering_excludes_short_sentences(summary: str):
    """
    **Validates: Requirements 3.1, 3.2**

    Property 6: For any summary string, every sentence in the output of
    generate_notes has length of at least 20 characters.
    """
    notes = generate_notes(summary)
    for note in notes:
        assert len(note) >= 20, (
            f"Note shorter than 20 chars found: {note!r} (len={len(note)})"
        )
