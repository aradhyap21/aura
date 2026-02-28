"""
AURA v2 — File Extractor
Extracts the COMPLETE raw text from PDF / DOCX / PPTX.
No truncation. No previews. Full document.
"""

import io
import re
import pdfplumber
import docx
from pptx import Presentation
from fastapi import HTTPException


# ─────────────────────────────────────────────────────────────────
# PDF
# ─────────────────────────────────────────────────────────────────

def extract_pdf(file_bytes: bytes) -> str:
    """
    Extract every page of a PDF.
    Handles multi-column, tables, and standard formatted PDFs.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text and text.strip():
                pages.append(text.strip())

    if not pages:
        raise HTTPException(
            status_code=422,
            detail=(
                "No text found in PDF. "
                "This is likely a scanned/image PDF. "
                "Please use a text-based PDF."
            )
        )
    return '\n\n'.join(pages)


# ─────────────────────────────────────────────────────────────────
# DOCX
# ─────────────────────────────────────────────────────────────────

def extract_docx(file_bytes: bytes) -> str:
    """
    Extract paragraphs + tables from a Word document.
    Preserves heading structure which is critical for topic detection.
    """
    doc   = docx.Document(io.BytesIO(file_bytes))
    parts = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        # Mark headings explicitly so topic detector can find them
        style = para.style.name.lower() if para.style else ""
        if "heading" in style:
            level = ''.join(filter(str.isdigit, style)) or '1'
            parts.append(f"{'#' * int(level)} {text}")
        else:
            parts.append(text)

    # Tables
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(' | '.join(cells))

    if not parts:
        raise HTTPException(status_code=422, detail="No text found in DOCX file.")

    return '\n'.join(parts)


# ─────────────────────────────────────────────────────────────────
# PPTX
# ─────────────────────────────────────────────────────────────────

def extract_pptx(file_bytes: bytes) -> str:
    """
    Extract text from every slide.
    Slide titles become topic markers.
    Body text becomes content.
    Speaker notes are included.
    """
    prs   = Presentation(io.BytesIO(file_bytes))
    parts = []

    for slide_num, slide in enumerate(prs.slides, start=1):
        slide_title   = ""
        slide_content = []

        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue

            shape_text = shape.text_frame.text.strip()
            if not shape_text:
                continue

            # Title placeholder → treat as heading
            if shape.shape_type == 13 or (hasattr(shape, "placeholder_format") and
               shape.placeholder_format is not None and
               shape.placeholder_format.idx == 0):
                slide_title = shape_text
            else:
                slide_content.append(shape_text)

        # Assemble slide
        if slide_title:
            parts.append(f"# {slide_title}")
        elif slide_content:
            parts.append(f"# Slide {slide_num}")

        if slide_content:
            parts.append('\n'.join(slide_content))

        # Speaker notes
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                parts.append(f"[Notes: {notes}]")

    if not parts:
        raise HTTPException(status_code=422, detail="No text found in PPTX file.")

    return '\n\n'.join(parts)


# ─────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {
    ".pdf" : "PDF",
    ".docx": "DOCX",
    ".doc" : "DOCX",
    ".pptx": "PPTX",
}

ALLOWED_MIME = {
    "application/pdf"                                                            : "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"   : "DOCX",
    "application/msword"                                                         : "DOCX",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation" : "PPTX",
    "application/vnd.ms-powerpoint"                                              : "PPTX",
}

MAX_MB = 30


def detect_type(content_type: str, filename: str) -> str:
    ftype = ALLOWED_MIME.get(content_type)
    if not ftype:
        for ext, t in ALLOWED_EXTENSIONS.items():
            if filename.lower().endswith(ext):
                ftype = t
                break
    if not ftype:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file. Upload a PDF, DOCX, or PPTX."
        )
    return ftype


def extract_full_text(file_bytes: bytes, file_type: str) -> str:
    """Route to the correct extractor and return complete raw text."""
    try:
        if file_type == "PDF":
            return extract_pdf(file_bytes)
        elif file_type == "DOCX":
            return extract_docx(file_bytes)
        elif file_type == "PPTX":
            return extract_pptx(file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {str(e)}")