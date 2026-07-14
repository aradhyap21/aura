"""
Property-based tests for pdf_extractor.py

**Validates: Requirements 1.2, 1.3**
"""

import io
import re
import struct
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pdf_extractor import clean_text, extract_text


@given(st.text())
@settings(max_examples=500)
def test_clean_text_removes_whitespace_artifacts(raw_text: str) -> None:
    """
    Property 2: Text cleaning removes whitespace artifacts.

    For any raw text string, clean_text returns a string that:
    1. Contains no consecutive whitespace (no double spaces, newlines, tabs)
    2. Contains no non-printable control characters
    3. Is stripped (no leading/trailing whitespace)

    **Validates: Requirements 1.3**
    """
    result = clean_text(raw_text)

    # 1. No consecutive whitespace (double spaces, newlines, tabs)
    assert "\n" not in result, f"Output contains newline: {repr(result)}"
    assert "\t" not in result, f"Output contains tab: {repr(result)}"
    assert "\r" not in result, f"Output contains carriage return: {repr(result)}"
    assert "  " not in result, f"Output contains consecutive spaces: {repr(result)}"

    # 2. No non-printable control characters
    for ch in result:
        cat = unicodedata.category(ch)
        assert cat[0] != "C" or ch == " ", (
            f"Output contains non-printable control character {repr(ch)} "
            f"(unicode category {cat}): {repr(result)}"
        )

    # 3. Output is stripped (no leading/trailing whitespace)
    assert result == result.strip(), (
        f"Output has leading/trailing whitespace: {repr(result)}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pdf(text: str) -> bytes:
    """Build a minimal valid PDF containing *text* using only stdlib."""
    # Encode the text for the PDF stream (escape parentheses and backslashes)
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream_content = f"BT /F1 12 Tf 50 700 Td ({safe}) Tj ET".encode("latin-1", errors="replace")
    stream_len = len(stream_content)

    body = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]\n"
        b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n\n"
        + b"4 0 obj\n<< /Length " + str(stream_len).encode() + b" >>\nstream\n"
        + stream_content + b"\nendstream\nendobj\n\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    )

    # Build a simple cross-reference table
    offsets = []
    pos = 0
    for line in body.split(b"\n"):
        if line.endswith(b"obj"):
            offsets.append(pos)
        pos += len(line) + 1  # +1 for the newline

    # Recompute offsets properly by scanning the raw bytes
    offsets = []
    for i in range(1, 6):
        marker = (f"{i} 0 obj\n").encode()
        idx = body.find(marker)
        if idx != -1:
            offsets.append(idx)

    xref_offset = len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()

    trailer = (
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF"
    )

    return body + xref + trailer


# ---------------------------------------------------------------------------
# Property 1: Text extraction produces non-empty output
# ---------------------------------------------------------------------------

# Printable ASCII text that pdfplumber can reliably encode in a Type1 font
_printable_ascii = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126, blacklist_characters="\\()"),
    min_size=1,
)


@given(_printable_ascii)
@settings(max_examples=50)
def test_extract_text_returns_non_empty_for_valid_pdf(text: str) -> None:
    """
    Property 1: Text extraction produces non-empty output.

    For any valid PDF bytes containing text, extract_text returns a non-empty string.

    **Validates: Requirements 1.2**
    """
    pdf_bytes = make_pdf(text)
    result = extract_text(pdf_bytes)
    assert isinstance(result, str), "extract_text must return a str"
    assert len(result.strip()) > 0, "extract_text must return a non-empty string for a valid PDF"
