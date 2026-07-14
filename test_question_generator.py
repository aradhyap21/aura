"""
Property-based tests for question_generator.py

Tests use hypothesis and mock the qg_pipeline to avoid loading the real T5 model.

Properties covered:
  - Property 9: Bloom levels cycle in round-robin order (Validates: Requirements 4.3)
  - Property 7: Question generator produces valid QAPairs (Validates: Requirements 4.1, 4.5, 4.8)
  - Property 8: T5 input is correctly formatted (Validates: Requirements 4.2)
  - Property 10: Question generation is bounded by 10 notes (Validates: Requirements 4.4)
"""

from unittest.mock import MagicMock

from hypothesis import given, settings
import hypothesis.strategies as st

from question_generator import generate_questions, extract_key_phrases

REALISTIC_NOTES = [
    "The mitochondria is the powerhouse of the cell and produces energy.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "The water cycle describes the continuous movement of water on Earth.",
    "Newton's laws of motion describe the relationship between force and acceleration.",
    "The human brain contains approximately 86 billion neurons.",
    "DNA carries the genetic information needed for growth and reproduction.",
    "The solar system consists of the sun and eight planets.",
    "Gravity is the force that attracts objects with mass toward each other.",
    "The periodic table organizes chemical elements by atomic number.",
    "Ecosystems consist of living organisms interacting with their environment.",
    "The immune system protects the body from harmful pathogens and diseases.",
    "Plate tectonics explains the movement of Earth's lithospheric plates.",
    "The speed of light in a vacuum is approximately 299 million meters per second.",
    "Cells are the basic structural and functional units of all living organisms.",
    "The carbon cycle describes the movement of carbon through the biosphere.",
]

notes_strategy = st.lists(st.sampled_from(REALISTIC_NOTES), min_size=1, max_size=20)
long_notes_strategy = st.lists(st.sampled_from(REALISTIC_NOTES), min_size=11, max_size=30)

BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze"]


def _make_mock_pipeline(question_text: str = "What is the answer?"):
    """Return a (mock_tokenizer, mock_model) tuple producing question_text."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {"input_ids": MagicMock()}
    mock_tokenizer.decode = MagicMock(return_value=question_text)

    mock_model = MagicMock()
    mock_model.generate = MagicMock(return_value=[[0]])

    return mock_tokenizer, mock_model


# ---------------------------------------------------------------------------
# Property 9: Bloom levels cycle in round-robin order
# ---------------------------------------------------------------------------

@given(notes_strategy)
@settings(max_examples=200)
def test_property9_bloom_levels_cycle_round_robin(notes):
    """Property 9: Bloom levels cycle in round-robin order. Validates: Requirements 4.3"""
    mock_pipeline = _make_mock_pipeline("What is the answer?")
    qa_pairs = generate_questions(notes, mock_pipeline, BLOOM_LEVELS)

    for i, pair in enumerate(qa_pairs):
        expected_level = BLOOM_LEVELS[i % len(BLOOM_LEVELS)]
        assert pair["bloom_level"] == expected_level


# ---------------------------------------------------------------------------
# Property 7: Question generator produces valid QAPairs
# ---------------------------------------------------------------------------

@given(notes_strategy)
@settings(max_examples=200)
def test_property7_generate_questions_produces_valid_qa_pairs(notes):
    """Property 7: Question generator produces valid QAPairs. Validates: Requirements 4.1, 4.5, 4.8"""
    mock_pipeline = _make_mock_pipeline("What is the answer?")
    qa_pairs = generate_questions(notes, mock_pipeline, BLOOM_LEVELS)

    notes_with_phrases = [n for n in notes[:10] if extract_key_phrases(n)]
    if notes_with_phrases:
        assert len(qa_pairs) >= 1

    for pair in qa_pairs:
        assert pair.get("question")
        assert pair.get("answer")
        assert pair.get("bloom_level")
        assert pair.get("source_note")
        assert "?" in pair["question"]
        assert pair["bloom_level"] in BLOOM_LEVELS


# ---------------------------------------------------------------------------
# Property 8: T5 input is correctly formatted
# ---------------------------------------------------------------------------

@given(notes_strategy)
@settings(max_examples=200)
def test_property8_t5_input_is_correctly_formatted(notes):
    """Property 8: T5 input is correctly formatted. Validates: Requirements 4.2"""
    mock_tokenizer, mock_model = _make_mock_pipeline("What is the answer?")
    generate_questions(notes, (mock_tokenizer, mock_model), BLOOM_LEVELS)

    for c in mock_tokenizer.call_args_list:
        t5_input = c.args[0]
        assert t5_input.startswith("Generate a question whose answer is: ")
        assert "Context: " in t5_input


# ---------------------------------------------------------------------------
# Property 10: Question generation is bounded by 10 notes
# ---------------------------------------------------------------------------

@given(long_notes_strategy)
@settings(max_examples=200)
def test_property10_question_generation_bounded_by_10_notes(notes):
    """Property 10: Question generation is bounded by 10 notes. Validates: Requirements 4.4"""
    assert len(notes) > 10
    mock_tokenizer, mock_model = _make_mock_pipeline("What is the answer?")
    qa_pairs = generate_questions(notes, (mock_tokenizer, mock_model), BLOOM_LEVELS)

    assert len(qa_pairs) <= 10
    assert mock_model.generate.call_count <= 10
