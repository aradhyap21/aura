"""
Property-based tests for summarizer.py

**Validates: Requirements 2.1, 2.2, 2.3**
"""

from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from summarizer import chunk_text, summarize


# ---------------------------------------------------------------------------
# Property 4: Chunks never exceed 1000 characters
# ---------------------------------------------------------------------------

@given(st.text())
@settings(max_examples=500)
def test_chunk_text_chunks_never_exceed_max_size(s: str) -> None:
    """
    Property 4: Chunks never exceed 1000 characters.
    **Validates: Requirements 2.2**
    """
    chunks = chunk_text(s, 1000)
    for chunk in chunks:
        assert len(chunk) <= 1000
    assert "".join(chunks) == s


# ---------------------------------------------------------------------------
# Helpers: mock (tokenizer, model) tuple
# ---------------------------------------------------------------------------

_SHORT_SUMMARY = "Short."


def _make_mock_summarizer(summary_text: str = _SHORT_SUMMARY):
    """Return a (mock_tokenizer, mock_model) tuple that produces summary_text."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.decode = MagicMock(return_value=summary_text)

    mock_model = MagicMock()
    mock_model.generate = MagicMock(return_value=[[0]])  # fake output_ids

    return mock_tokenizer, mock_model


_long_text = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=100,
).filter(lambda s: len(s.strip()) >= 50)


# ---------------------------------------------------------------------------
# Property 3: Summary is shorter than input
# ---------------------------------------------------------------------------

@given(_long_text)
@settings(max_examples=200)
def test_summarize_returns_shorter_than_input(text: str) -> None:
    """
    Property 3: Summary is shorter than input.
    **Validates: Requirements 2.1**
    """
    mock_summarizer = _make_mock_summarizer(_SHORT_SUMMARY)
    result = summarize(text, mock_summarizer)
    assert isinstance(result, str)
    assert len(result) < len(text)


# ---------------------------------------------------------------------------
# Property 5: Short chunks are skipped
# ---------------------------------------------------------------------------

_short_chunk = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    max_size=49,
).filter(lambda s: len(s.strip()) < 50)


@given(_short_chunk)
@settings(max_examples=300)
def test_summarize_skips_short_chunks(chunk: str) -> None:
    """
    Property 5: Short chunks are skipped.
    **Validates: Requirements 2.3**
    """
    mock_tokenizer, mock_model = _make_mock_summarizer()
    summarize(chunk, (mock_tokenizer, mock_model))
    mock_model.generate.assert_not_called()
