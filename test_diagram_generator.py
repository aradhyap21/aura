"""
Property-based tests for diagram_generator.py

Properties covered:
  - Property 13: Diagram output is valid Graphviz DOT syntax
    (Validates: Requirements 6.1, 6.2, 6.4)
"""

from hypothesis import given, settings
import hypothesis.strategies as st

from diagram_generator import generate_diagram


@given(st.lists(st.text(min_size=1), min_size=1))
@settings(max_examples=200)
def test_property13_diagram_valid_graphviz_syntax(notes):
    """
    Property 13: Diagram output is valid Graphviz DOT syntax.

    For any non-empty list of notes, generate_diagram returns a string that:
    1. Starts with "digraph G {"
    2. Contains at most 5 "->" edges (for 6 nodes)
    3. Is a non-empty string

    **Validates: Requirements 6.1, 6.2, 6.4**
    """
    result = generate_diagram(notes)

    assert isinstance(result, str) and len(result) > 0
    assert result.strip().startswith("digraph G {"), (
        f"Output must start with 'digraph G {{', got: {result[:60]!r}"
    )
    arrow_count = result.count("->")
    assert arrow_count <= 5, (
        f"Expected at most 5 '->' edges (for 6 nodes), got {arrow_count}"
    )
