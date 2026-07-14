"""
AURA v3 — Analyze Service
Coordinator between API, extractor, and semantic topic extractor.
"""

from utils.extractor import extract_full_text, detect_file_type
from utils.semantic_topic_extractor import parse_topics
from models.schemas import ExtractionResult


async def analyze_document(file) -> ExtractionResult:

    # Step 1 — detect file type
    file_type = detect_file_type(file.content_type, file.filename)

    # Step 2 — read and size check
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)
    if size_mb > 30:
        raise ValueError(f"File too large ({size_mb:.1f}MB). Max 30MB.")

    # Step 3 — extract full text
    print(f"  Extracting text from {file_type.upper()}...")
    raw_text = extract_full_text(file_bytes, file_type)

    # Step 4 — warnings
    word_count = len(raw_text.split())
    warnings   = []
    if word_count < 100:
        warnings.append("Very little text extracted. Check if file is a scanned image.")
    if word_count > 150_000:
        warnings.append("Very large document. Processing may take longer.")

    # Step 5 — semantic topic extraction
    print(f"  Running semantic topic extraction on {word_count} words...")
    topics = parse_topics(raw_text)

    # Step 6 — assemble result
    high   = sum(1 for t in topics if t.priority.value == "High")
    medium = sum(1 for t in topics if t.priority.value == "Medium")
    low    = sum(1 for t in topics if t.priority.value == "Low")
    subs   = sum(len(t.subtopics) for t in topics)

    return ExtractionResult(
    file_name       = file.filename,
    file_type       = file_type,
    total_words     = word_count,
    total_topics    = len(topics),
    high_count      = high,
    medium_count    = medium,
    low_count       = low,
    total_subtopics = subs,
    topics          = topics,
    warnings        = warnings,
)