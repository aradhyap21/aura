"""
MCQ Builder: wraps QA pairs into multiple-choice questions with distractors.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import random


def generate_distractors(answer: str, all_answers: list[str], n: int = 3) -> list[str]:
    """Pick n distractor answers from the answer pool, excluding the correct one."""
    pool = [a for a in all_answers if a != answer]
    # If pool has fewer than n items, sample as many as available
    k = min(n, len(pool))
    return random.sample(pool, k)


def build_mcqs(qa_pairs: list[dict]) -> list[dict]:
    """Build MCQ dicts with shuffled options and correct index."""
    all_answers = [q["answer"] for q in qa_pairs]
    mcqs = []

    for item in qa_pairs:
        correct = item["answer"]
        distractors = generate_distractors(correct, all_answers, n=3)
        options = distractors + [correct]
        random.shuffle(options)
        correct_index = options.index(correct)

        mcqs.append({
            "question": item["question"],
            "options": options,
            "correct_index": correct_index,
            "bloom_level": item["bloom_level"],
        })

    return mcqs
