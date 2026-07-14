"""
AURA — Dataset Cleaner
Removes bad examples from aura_training_final.jsonl before retraining.

Filters out:
- Placeholder outputs (from old formatter)
- Outputs that are just copies of input
- Too short or too long outputs
- Low quality outputs

Run:
    python clean_dataset.py
"""

import json
import re
import os

INPUT_PATH  = "data/aura_training_final.jsonl"
OUTPUT_PATH = "data/aura_training_clean.jsonl"
STATS_PATH  = "data/clean_stats.json"

# Phrases that indicate a placeholder / bad output
BAD_PHRASES = [
    "[explanation to be filled]",
    "[bullets to be filled]",
    "[analogy to be filled]",
    "to be filled",
    "placeholder",
    "[insert",
    "n/a",
    "none",
    "unknown",
    "not available",
    "see above",
    "see below",
    "as mentioned",
    "as stated",
]

def is_bad_output(output: str, input_text: str) -> bool:
    if not output or not output.strip():
        return True

    output_lower = output.lower().strip()
    words        = output.split()

    # Too short
    if len(words) < 8:
        return True

    # Too long
    if len(words) > 500:
        return True

    # Contains placeholder phrases
    for phrase in BAD_PHRASES:
        if phrase in output_lower:
            return True

    # Output is just copying input (>80% overlap)
    input_words  = set(re.findall(r'\b\w{4,}\b', input_text.lower()))
    output_words = set(re.findall(r'\b\w{4,}\b', output_lower))
    if input_words and len(output_words) > 0:
        overlap = len(input_words & output_words) / len(output_words)
        if overlap > 0.85:
            return True

    # Output is just a fragment ending mid-sentence and too short
    if len(words) < 15 and not output.strip().endswith(('.', '!', '?')):
        return True

    return False


def clean():
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found. Run build_dataset.py first.")
        return

    total    = 0
    kept     = 0
    removed  = 0
    by_reason = {
        'placeholder': 0,
        'too_short'  : 0,
        'too_long'   : 0,
        'copy_input' : 0,
        'fragment'   : 0,
    }

    source_counts = {}

    print(f"Cleaning {INPUT_PATH}...")

    with open(INPUT_PATH, 'r', encoding='utf-8') as fin, \
         open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                ex     = json.loads(line)
                output = ex.get('output', '')
                inp    = ex.get('input', '')

                if is_bad_output(output, inp):
                    removed += 1
                    continue

                fout.write(json.dumps(ex, ensure_ascii=False) + '\n')
                kept += 1

                src = ex.get('source', 'unknown')
                source_counts[src] = source_counts.get(src, 0) + 1

            except Exception:
                removed += 1
                continue

    stats = {
        'total'        : total,
        'kept'         : kept,
        'removed'      : removed,
        'kept_pct'     : round(kept / max(total, 1) * 100, 1),
        'by_source'    : source_counts,
    }

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Cleaning Complete")
    print(f"{'='*50}")
    print(f"  Total examples : {total}")
    print(f"  Kept           : {kept} ({stats['kept_pct']}%)")
    print(f"  Removed        : {removed}")
    print(f"\n  By source:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<20} {count}")
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"{'='*50}")


if __name__ == '__main__':
    clean()