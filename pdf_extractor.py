import io
import re
import unicodedata

import pdfplumber


def extract_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)

    raw_text = "\n".join(pages_text)

    if not raw_text.strip():
        raise ValueError(
            "No extractable text found in PDF. "
            "The file may be a scanned image or contain no selectable text."
        )

    return raw_text


def clean_text(raw_text: str) -> str:
    """Remove excess whitespace, line breaks, and non-printable characters."""
    # Strip non-printable / control characters (keep normal whitespace for now)
    cleaned = "".join(
        ch for ch in raw_text
        if unicodedata.category(ch)[0] != "C" or ch in (" ", "\t", "\n", "\r")
    )

    # Collapse all whitespace (spaces, tabs, newlines) into a single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()
