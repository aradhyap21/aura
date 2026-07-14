"""
NotesService: AI-generated structured notes for any topic using NVIDIA NIM API.
Produces: Summary, Key Concepts, Definitions, Examples, Exam Revision block.
"""

import re
from openai import OpenAI

_CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)


def _call_model(prompt: str, max_tokens: int = 2500) -> str:
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


def generate_topic_notes(
    topic: str,
    level: str = "intermediate",
    learning_goal: str = "",
) -> dict:
    """
    Generate structured exam-oriented notes for a given topic.

    Returns dict with keys:
      title, summary, key_points, definitions, examples,
      formulae, important_concepts, revision_block, raw
    """
    goal_line = f"Learning goal: {learning_goal}" if learning_goal else ""

    prompt = f"""You are an expert academic tutor. Generate comprehensive, exam-oriented study notes.

Topic: {topic}
Level: {level}
{goal_line}

Use EXACTLY these section headers on their own line:

SECTION A: SUMMARY
SECTION B: KEY CONCEPTS
SECTION C: DEFINITIONS
SECTION D: EXAMPLES
SECTION E: EXAM REVISION

SECTION A: SUMMARY
Write 3-5 detailed paragraphs explaining the topic thoroughly. Cover background, core ideas, and significance.

SECTION B: KEY CONCEPTS
Write 8-12 bullet points. Each bullet = one key concept with a brief explanation.

SECTION C: DEFINITIONS
Write 6-10 short, memory-friendly definitions. Format: **Term**: definition

SECTION D: EXAMPLES
Write 3-5 concrete examples or worked problems that illustrate the concepts.

SECTION E: EXAM REVISION
Write a 1-minute quick revision block: 5-8 bullet points of the most important facts to remember for exams.
"""

    try:
        raw = _call_model(prompt, max_tokens=2500)
        return _parse_notes(raw, topic)
    except Exception as e:
        return {
            "title": topic, "summary": "", "key_points": "",
            "definitions": "", "examples": "", "formulae": "",
            "important_concepts": "", "revision_block": "",
            "raw": "", "error": str(e),
        }


_SEC_RE = re.compile(
    r"SECTION\s+[A-E]:\s*(SUMMARY|KEY CONCEPTS|DEFINITIONS|EXAMPLES|EXAM REVISION)\s*\n",
    re.IGNORECASE,
)


def _parse_notes(text: str, topic: str) -> dict:
    result = {
        "title": topic,
        "summary": "", "key_points": "", "definitions": "",
        "examples": "", "revision_block": "", "raw": text,
    }
    parts = _SEC_RE.split(text)
    i = 1
    while i < len(parts) - 1:
        header = parts[i].upper().strip()
        content = parts[i + 1].strip()
        if "SUMMARY" in header:
            result["summary"] = content
        elif "KEY" in header:
            result["key_points"] = content
        elif "DEFINITION" in header:
            result["definitions"] = content
        elif "EXAMPLE" in header:
            result["examples"] = content
        elif "REVISION" in header:
            result["revision_block"] = content
        i += 2
    return result
