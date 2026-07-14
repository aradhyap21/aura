"""
QuestionService: AI-generated Bloom-taxonomy questions, MCQs, and active recall
questions for any topic at any difficulty level.
"""

import re
from openai import OpenAI
from adaptive_engine import BLOOM_LEVELS

_CLIENT = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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",
)


def _call_model(prompt: str, max_tokens: int = 2000) -> str:
    completion = _CLIENT.chat.completions.create(
        model="meta/llama-3.1-8b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
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


def generate_bloom_questions(topic: str, bloom_level: int, count: int = 5) -> list[dict]:
    """Generate Bloom-taxonomy questions for a topic at a specific level."""
    level_name = BLOOM_LEVELS.get(bloom_level, "Remember")
    prompt = f"""Generate exactly {count} exam questions for the topic "{topic}" at Bloom's Taxonomy level: {level_name}.

Level descriptions:
- Remember: recall facts, definitions, lists
- Understand: explain, describe, summarize
- Apply: use knowledge in new situations, solve problems
- Analyze: compare, contrast, break down, examine
- Evaluate: judge, critique, justify, recommend

Format each question exactly like this:
Q: [question text]
TYPE: {level_name}
ANSWER: [expected answer in 1-3 sentences]

Generate all {count} questions now.
"""
    try:
        text = _call_model(prompt, max_tokens=1500)
        return _parse_bloom_questions(text, bloom_level)
    except Exception:
        return []


def generate_mcqs(topic: str, bloom_level: int, count: int = 5) -> list[dict]:
    """Generate MCQs for a topic at a specific Bloom level."""
    level_name = BLOOM_LEVELS.get(bloom_level, "Remember")
    prompt = f"""Generate exactly {count} multiple-choice questions for "{topic}" at Bloom level: {level_name}.

Use EXACTLY this format for each question:

Q: [question]
A) [option]
B) [option]
C) [option]
D) [option]
ANSWER: [A/B/C/D]
EXPLANATION: [one sentence]

Generate all {count} now.
"""
    try:
        text = _call_model(prompt, max_tokens=2000)
        return _parse_mcqs(text, bloom_level)
    except Exception:
        return []


def generate_recall_questions(topic: str, count: int = 5) -> list[dict]:
    """Generate free-text active recall questions."""
    prompt = f"""Generate exactly {count} active recall questions for "{topic}".

These should test memory retrieval without multiple choice options.
Focus on key facts, processes, and concepts.

Format each question exactly like this:
Q: [question]
EXPECTED: [model answer in 2-3 sentences]

Generate all {count} now.
"""
    try:
        text = _call_model(prompt, max_tokens=1500)
        return _parse_recall_questions(text)
    except Exception:
        return []


def _parse_bloom_questions(text: str, bloom_level: int) -> list[dict]:
    questions = []
    blocks = re.split(r"\n(?=Q:)", text.strip())
    for block in blocks:
        q = re.search(r"Q:\s*(.+)", block)
        a = re.search(r"ANSWER:\s*(.+)", block, re.DOTALL)
        if q:
            questions.append({
                "question": q.group(1).strip(),
                "answer": a.group(1).strip() if a else "",
                "bloom_level": bloom_level,
                "type": "bloom",
            })
    return questions


def _parse_mcqs(text: str, bloom_level: int) -> list[dict]:
    mcqs = []
    blocks = re.split(r"\n(?=Q:)", text.strip())
    for block in blocks:
        q = re.search(r"Q:\s*(.+)", block)
        a = re.search(r"A\)\s*(.+)", block)
        b = re.search(r"B\)\s*(.+)", block)
        c = re.search(r"C\)\s*(.+)", block)
        d = re.search(r"D\)\s*(.+)", block)
        ans = re.search(r"ANSWER:\s*([ABCD])", block, re.IGNORECASE)
        exp = re.search(r"EXPLANATION:\s*(.+)", block, re.DOTALL)
        if not (q and a and b and c and d and ans):
            continue
        options = [x.group(1).strip() for x in [a, b, c, d]]
        correct_index = {"A": 0, "B": 1, "C": 2, "D": 3}.get(ans.group(1).upper(), 0)
        mcqs.append({
            "question": q.group(1).strip(),
            "options": options,
            "correct_index": correct_index,
            "explanation": exp.group(1).strip() if exp else "",
            "bloom_level": bloom_level,
            "type": "mcq",
        })
    return mcqs


def _parse_recall_questions(text: str) -> list[dict]:
    questions = []
    blocks = re.split(r"\n(?=Q:)", text.strip())
    for block in blocks:
        q = re.search(r"Q:\s*(.+)", block)
        e = re.search(r"EXPECTED:\s*(.+)", block, re.DOTALL)
        if q:
            questions.append({
                "question": q.group(1).strip(),
                "expected": e.group(1).strip() if e else "",
                "type": "recall",
            })
    return questions
