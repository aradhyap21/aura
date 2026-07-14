"""
AURA v2 — Analysis Service
Wires together: file extraction → topic parsing → result assembly
"""

from fastapi import UploadFile, HTTPException
from utils.extractor   import detect_type, extract_full_text
from utils.topic_parser import parse_topics
from models.schemas    import ExtractionResult, Priority


MAX_MB = 30


async def analyze(file: UploadFile) -> ExtractionResult:
    """
    Complete pipeline:
    1. Detect and validate file type
    2. Read full file bytes
    3. Extract complete text (no truncation)
    4. Parse topics and subtopics from structure
    5. Score and classify all topics
    6. Return structured result
    """

    # ── 1. File type ─────────────────────────────────────────────
    file_type = detect_type(file.content_type or "", file.filename or "")

    # ── 2. Read file ─────────────────────────────────────────────
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)

    if size_mb > MAX_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {MAX_MB} MB."
        )

    # ── 3. Extract FULL text ─────────────────────────────────────
    raw_text = extract_full_text(file_bytes, file_type)

    total_words = len(raw_text.split())
    warnings    = []

    if total_words < 100:
        warnings.append(
            "Very few words extracted. File may be a scanned image or mostly visual content."
        )
    if total_words > 150_000:
        warnings.append("Very large document. Processing may take a few seconds.")

    # ── 4. Parse topics ──────────────────────────────────────────
    topics = parse_topics(raw_text)

    if not topics:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not detect any topics or structure in this document. "
                "Try a document with clear headings or sections."
            )
        )

    # ── 5. Count by priority ─────────────────────────────────────
    high_count   = sum(1 for t in topics if t.priority == Priority.HIGH)
    medium_count = sum(1 for t in topics if t.priority == Priority.MEDIUM)
    low_count    = sum(1 for t in topics if t.priority == Priority.LOW)
    total_subs   = sum(len(t.subtopics) for t in topics)

    # ── 6. Return result ─────────────────────────────────────────
    return ExtractionResult(
        file_name       = file.filename or "unknown",
        file_type       = file_type,
        total_words     = total_words,
        total_topics    = len(topics),
        total_subtopics = total_subs,
        high_count      = high_count,
        medium_count    = medium_count,
        low_count       = low_count,
        warnings        = warnings,
        topics          = topics,
    )