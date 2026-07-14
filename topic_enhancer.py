"""
Topic-aware document processing using NVIDIA NIM API (Llama 3.1).
Processes the full document in chunks, identifies all topics, and generates
structured notes and a comprehensive summary with full content coverage.
"""

import re
from openai import OpenAI

_CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)

_CHUNK_SIZE = 4000
_MAX_CHUNKS = 5


def _call_model(prompt: str, max_tokens: int = 2000) -> str:
    completion = _CLIENT.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        top_p=0.9,
        max_tokens=max_tokens,
        stream=True,
    )
    result = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        if chunk.choices[0].delta.content is not None:
            result.append(chunk.choices[0].delta.content)
    return "".join(result)


def _split_into_chunks(text: str) -> list[str]:
    """Split document at paragraph boundaries into chunks of ~4000 chars."""
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > _CHUNK_SIZE:
            if current.strip():
                chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks[:_MAX_CHUNKS]


def _process_chunk(chunk: str, chunk_num: int, total: int) -> dict:
    """Extract topics and notes from a single chunk."""
    prompt = f"""You are an academic assistant analyzing part {chunk_num} of {total} of a document.

Extract ALL topics from this section. For each topic:
- Write the topic name as a bold heading: **Topic Name**
- Write 5-8 detailed bullet points covering key facts, definitions, and concepts

Then write a SUMMARY paragraph (4-6 sentences) covering all content in this section.

Use this exact format:
TOPICS
**Topic 1**
- bullet
- bullet

**Topic 2**
- bullet

SUMMARY
[paragraph]

Document section:
{chunk}
"""
    try:
        text = _call_model(prompt, max_tokens=1500)
        topics_match = re.search(r"TOPICS\s*\n(.*?)(?=SUMMARY|\Z)", text, re.DOTALL)
        summary_match = re.search(r"SUMMARY\s*\n(.*)", text, re.DOTALL)
        return {
            "topics": topics_match.group(1).strip() if topics_match else text,
            "summary": summary_match.group(1).strip() if summary_match else "",
        }
    except Exception as e:
        return {"topics": "", "summary": "", "error": str(e)}


def _synthesize_final_output(all_topics: str, all_summaries: str) -> dict:
    """Combine all chunk outputs into a final coherent summary and notes."""
    prompt = f"""You are an academic assistant. Below are topic notes and summaries extracted from different sections of a document.

Combine them into:
1. A FINAL SUMMARY: comprehensive, topic-wise, covering ALL content. Use bold headings per topic. 4-6 sentences per topic.
2. FINAL NOTES: consolidated topic-wise bullet notes. Bold heading per topic. 6-10 bullets per topic. No duplication.

Use exactly these headers:
FINAL SUMMARY
FINAL NOTES

All topics extracted:
{all_topics[:6000]}

All section summaries:
{all_summaries[:3000]}
"""
    try:
        text = _call_model(prompt, max_tokens=3000)
        summary_match = re.search(r"FINAL SUMMARY\s*\n(.*?)(?=FINAL NOTES|\Z)", text, re.DOTALL)
        notes_match = re.search(r"FINAL NOTES\s*\n(.*)", text, re.DOTALL)
        return {
            "summary": summary_match.group(1).strip() if summary_match else text[:2000],
            "notes": notes_match.group(1).strip() if notes_match else "",
        }
    except Exception as e:
        return {"summary": "", "notes": "", "error": str(e)}


def process_document_with_topics(text: str) -> dict:
    """
    Main entry point. Takes full extracted document text, processes it
    chunk by chunk to identify all topics, then synthesizes a final
    comprehensive summary and structured notes.

    Returns dict with keys: 'summary', 'notes', 'error' (if any).
    Falls back gracefully — never crashes the app.
    """
    if not text or not text.strip():
        return {"summary": "", "notes": "", "error": "Empty document text"}

    chunks = _split_into_chunks(text)
    if not chunks:
        return {"summary": "", "notes": "", "error": "Could not split document"}

    all_topics_parts = []
    all_summary_parts = []

    for i, chunk in enumerate(chunks):
        result = _process_chunk(chunk, i + 1, len(chunks))
        if result.get("topics"):
            all_topics_parts.append(result["topics"])
        if result.get("summary"):
            all_summary_parts.append(result["summary"])

    if not all_topics_parts:
        return {"summary": "", "notes": "", "error": "No topics extracted from document"}

    # If only one chunk, use it directly without a synthesis call
    if len(chunks) == 1:
        return {
            "summary": all_summary_parts[0] if all_summary_parts else "",
            "notes": all_topics_parts[0] if all_topics_parts else "",
        }

    # Multiple chunks: synthesize into a coherent final output
    all_topics = "\n\n".join(all_topics_parts)
    all_summaries = "\n\n".join(all_summary_parts)
    return _synthesize_final_output(all_topics, all_summaries)