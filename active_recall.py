"""
Active Recall Builder

Builds interactive recall prompts from notes and QA pairs, and evaluates
user answers using token-overlap (Jaccard) scoring.
"""

from models import RecallPrompt


def evaluate_answer(user_answer: str, expected_answer: str) -> dict:
    """Score user answer against expected using simple token overlap (Jaccard similarity).

    Postconditions:
    - Returns dict with keys: score (float 0.0–1.0), is_correct (bool), expected (str)
    - score == 1.0 if and only if token sets are identical (case-insensitive)
    - is_correct is True when score >= 0.5
    """
    user_tokens = set(user_answer.lower().split())
    expected_tokens = set(expected_answer.lower().split())

    if not user_tokens and not expected_tokens:
        score = 1.0
    else:
        intersection = user_tokens & expected_tokens
        union = user_tokens | expected_tokens
        score = len(intersection) / len(union) if union else 1.0

    return {
        "score": score,
        "is_correct": score >= 0.5,
        "expected": expected_answer,
    }


def _make_cloze_prompt(note: str) -> dict | None:
    """Create a cloze prompt from a note by blanking out the first word/phrase.

    Returns a dict with prompt, expected, prompt_type, or None if note is empty.
    """
    words = note.split()
    if not words:
        return None

    # Blank out the first word as the key phrase
    key_phrase = words[0]
    rest = " ".join(words[1:])
    prompt = f"___ {rest}" if rest else "___"

    return {
        "prompt": prompt,
        "expected": key_phrase,
        "prompt_type": "cloze",
    }


def build_recall_prompts(
    notes: list[str],
    qa_pairs: list[dict],
) -> list[dict]:
    """Combine note-based fill-in-the-blank prompts with QA-based prompts.

    Creates:
    - QA prompts: question → expected answer, prompt_type="qa"
    - Cloze prompts: blank out first word of each note, prompt_type="cloze"

    Every returned dict has: prompt, expected, prompt_type fields.
    prompt_type is either "qa" or "cloze".
    """
    prompts = []

    # QA-based prompts
    for pair in qa_pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        if question and answer:
            prompts.append({
                "prompt": question,
                "expected": answer,
                "prompt_type": "qa",
            })

    # Cloze prompts from notes
    for note in notes:
        cloze = _make_cloze_prompt(note)
        if cloze is not None:
            prompts.append(cloze)

    return prompts
