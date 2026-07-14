"""
Diagram Generator: converts notes into a Graphviz digraph string.
"""

import re

_MAX_LABEL_LEN = 35
_CLEAN_RE = re.compile(r'["\\\[\]{}<>]')


def _clean_label(text: str) -> str:
    """Truncate and strip special characters from a node label."""
    text = _CLEAN_RE.sub("", text)
    return text[:_MAX_LABEL_LEN].strip()


def generate_diagram(notes: list[str], max_nodes: int = 6) -> str:
    """Convert notes into a Graphviz digraph DOT string.

    Returns a valid DOT string renderable by st.graphviz_chart().
    """
    nodes = notes[:max_nodes]
    if not nodes:
        return 'digraph G { node0 [label="No notes available"]; }'

    lines = [
        "digraph G {",
        "    rankdir=LR;",
        '    node [shape=box style=filled fillcolor=lightblue fontname="Helvetica"];',
    ]

    for i, note in enumerate(nodes):
        label = _clean_label(note)
        lines.append(f'    node{i} [label="{label}"];')

    for i in range(len(nodes) - 1):
        lines.append(f"    node{i} -> node{i + 1};")

    lines.append("}")
    return "\n".join(lines)
