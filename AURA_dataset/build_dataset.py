"""
AURA — Unified Dataset Builder
================================
Downloads and converts 4 sources into AURA training format:

1. ELI5        — 270k Reddit explanations (explain like I'm 5)
2. Dolly 15k   — 15k human-written instruction-response pairs
3. FLAN        — Google's instruction tuning dataset
4. Your own    — wikipedia_raw.json (975 topics already collected)

Output: data/aura_training_final.jsonl
Format: Alpaca instruction format (same as before)

Run:
    pip install datasets tqdm
    python build_dataset.py
"""

import json
import os
import re
import random
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────
OUTPUT_PATH      = "data/aura_training_final.jsonl"
STATS_PATH       = "data/build_stats.json"
WIKI_RAW_PATH    = "data/wikipedia_raw.json"
EXISTING_PATH    = "data/aura_dataset.jsonl"
ANNOTATED_PATH   = "data/annotated_dataset.jsonl"

# How many examples to take from each source
# Keeping balanced so no single source dominates
LIMITS = {
    "eli5"       : 8000,
    "dolly"      : 5000,
    "flan"       : 5000,
    "wikipedia"  : 2697,   # all of them
    "annotated"  : 99999,  # all annotated — highest quality, keep all
}

# ── AURA format ───────────────────────────────────────────────────
INSTRUCTIONS = {
    "explanation": "You are an expert tutor. Explain this topic simply and clearly for a student who is studying for an exam.",
    "bullets"    : "You are an expert tutor. Generate exactly 5 concise exam bullet points covering the most important facts about this topic.",
    "analogy"    : "You are an expert tutor. Create a memorable real-world analogy that helps a student understand this topic intuitively.",
    "qa"         : "You are an expert tutor. Answer this question clearly and accurately for a student.",
    "summary"    : "You are an expert tutor. Summarize this content clearly for a student.",
}

def make_example(instruction_key, input_text, output_text, topic="", source=""):
    return {
        "instruction" : INSTRUCTIONS.get(instruction_key, instruction_key),
        "input"       : input_text.strip()[:1500],
        "output"      : output_text.strip(),
        "task"        : instruction_key,
        "topic"       : topic,
        "source"      : source,
        "annotated"   : source == "annotated",
    }


# ── Cleaning helpers ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove Reddit formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    return text.strip()


def is_good_output(text: str, min_words: int = 20, max_words: int = 400) -> bool:
    """Filter out outputs that are too short, too long, or low quality."""
    if not text:
        return False
    words = text.split()
    if len(words) < min_words or len(words) > max_words:
        return False
    # Skip outputs that are mostly links or code
    if text.count('http') > 2:
        return False
    if text.count('```') > 1:
        return False
    return True


# ═══════════════════════════════════════════════════════════════════
# SOURCE 1 — ELI5
# ═══════════════════════════════════════════════════════════════════

def load_eli5(limit: int) -> list:
    """
    Replaces ELI5 (deprecated) with 2 working alternatives:
    1. OpenHermes — high quality explanation pairs
    2. SQuAD v2 — Stanford QA dataset
    """
    examples = []

    # Source A: OpenHermes
    print("\n📚 Loading OpenHermes (explanation pairs)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
        skipped = 0
        checked = 0
        MAX_CHECK = limit * 4

        for item in tqdm(ds, desc="  OpenHermes", total=MAX_CHECK):
            if len(examples) >= limit // 2 or checked >= MAX_CHECK:
                break
            checked += 1
            try:
                convs = item.get('conversations', [])
                if len(convs) < 2:
                    skipped += 1
                    continue
                human = next((c['value'] for c in convs if c.get('from') == 'human'), '')
                asst  = next((c['value'] for c in convs if c.get('from') == 'gpt'), '')
                human = clean_text(human)
                asst  = clean_text(asst)
                if not human or not is_good_output(asst, min_words=20, max_words=350):
                    skipped += 1
                    continue
                h_lower = human.lower()
                if not any(kw in h_lower for kw in ['explain', 'what is', 'how does', 'describe', 'define', 'why']):
                    skipped += 1
                    continue
                examples.append(make_example(
                    instruction_key = "explanation",
                    input_text      = human,
                    output_text     = asst,
                    topic           = human[:80],
                    source          = "openhermes",
                ))
            except Exception:
                skipped += 1
        print(f"  ✅ {len(examples)} from OpenHermes ({skipped} skipped)")
    except Exception as e:
        print(f"  ⚠ OpenHermes failed: {e}")

    # Source B: SQuAD v2
    print("\n📚 Loading SQuAD v2...")
    try:
        from datasets import load_dataset
        ds      = load_dataset("rajpurkar/squad_v2", split="train")
        skipped = 0
        for item in tqdm(ds, desc="  SQuAD"):
            if len(examples) >= limit:
                break
            try:
                question = clean_text(item.get('question', ''))
                context  = clean_text(item.get('context', ''))
                answers  = item.get('answers', {}).get('text', [])
                if not answers or not question or not context:
                    skipped += 1
                    continue
                answer = clean_text(answers[0])
                if not is_good_output(answer, min_words=5, max_words=200):
                    skipped += 1
                    continue
                examples.append(make_example(
                    instruction_key = "qa",
                    input_text      = f"Question: {question}\n\nContext: {context[:600]}",
                    output_text     = answer,
                    topic           = question[:80],
                    source          = "squad",
                ))
            except Exception:
                skipped += 1
        print(f"  ✅ {len(examples)} total after SQuAD ({skipped} skipped)")
    except Exception as e:
        print(f"  ⚠ SQuAD failed: {e}")

    return examples[:limit]

# ═══════════════════════════════════════════════════════════════════
# SOURCE 2 — DOLLY 15K
# ═══════════════════════════════════════════════════════════════════

def load_dolly(limit: int) -> list:
    """
    Databricks Dolly 15k
    Human-written instruction-response pairs.
    Filter for explanation, summarization, and QA categories.
    """
    print("\n📚 Loading Dolly 15k...")
    try:
        from datasets import load_dataset
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"  ✗ Dolly failed: {e}")
        return []

    # Categories relevant to AURA
    KEEP_CATEGORIES = {
        'open_qa', 'closed_qa', 'summarization',
        'information_extraction', 'general_qa'
    }

    examples = []
    skipped  = 0

    for item in tqdm(ds, desc="  Dolly"):
        if len(examples) >= limit:
            break

        try:
            category = item.get('category', '')
            if category not in KEEP_CATEGORIES:
                skipped += 1
                continue

            instruction = clean_text(item.get('instruction', ''))
            context     = clean_text(item.get('context', ''))
            response    = clean_text(item.get('response', ''))

            if not instruction or not is_good_output(response, min_words=20):
                skipped += 1
                continue

            # Build input
            if context:
                input_text = f"{instruction}\n\nContext: {context}"
            else:
                input_text = instruction

            # Map category to AURA task
            task = "explanation" if "explain" in instruction.lower() else "qa"
            if category == "summarization":
                task = "summary"

            examples.append(make_example(
                instruction_key = task,
                input_text      = input_text,
                output_text     = response,
                topic           = instruction[:80],
                source          = "dolly",
            ))

        except Exception:
            skipped += 1
            continue

    print(f"  ✅ {len(examples)} examples ({skipped} skipped)")
    return examples


# ═══════════════════════════════════════════════════════════════════
# SOURCE 3 — FLAN
# ═══════════════════════════════════════════════════════════════════

def load_flan(limit: int) -> list:
    """
    FLAN — Google's instruction tuning dataset
    Filter for explanation and summarization tasks only.
    """
    print("\n📚 Loading FLAN...")
    try:
        from datasets import load_dataset
        # Use the smaller, cleaner FLAN subset
        ds = load_dataset(
            "Muennighoff/flan",
            split      = "train",
            streaming  = True,  # stream — don't download all 50GB
        )
    except Exception as e:
        print(f"  ✗ FLAN failed: {e}")
        return []

    # Keywords that indicate explanation/education tasks
    EXPLANATION_KEYWORDS = [
        'explain', 'what is', 'describe', 'define', 'summarize',
        'summary', 'how does', 'why does', 'what does', 'meaning of',
    ]

    examples = []
    skipped  = 0
    checked  = 0
    MAX_CHECK = limit * 15  # check up to 8x limit to find good ones

    for item in tqdm(ds, desc="  FLAN", total=MAX_CHECK):
        if len(examples) >= limit or checked >= MAX_CHECK:
            break
        checked += 1

        try:
            inputs  = clean_text(item.get('inputs', ''))
            targets = clean_text(item.get('targets', ''))

            if not inputs or not is_good_output(targets, min_words=15, max_words=300):
                skipped += 1
                continue

            # Skip if it's a multiple choice question
            if 'OPTIONS:' in inputs or inputs[:20].startswith('Answer:'):
                skipped += 1
                continue

            inputs_lower = inputs.lower()
            task = "summary" if "summarize" in inputs_lower or "summary" in inputs_lower else "explanation"

            examples.append(make_example(
                instruction_key = task,
                input_text      = inputs,
                output_text     = targets,
                topic           = inputs[:80],
                source          = "flan",
            ))

        except Exception:
            skipped += 1
            continue

    print(f"  ✅ {len(examples)} examples ({skipped} skipped)")
    return examples


# ═══════════════════════════════════════════════════════════════════
# SOURCE 4 — YOUR WIKIPEDIA DATA
# ═══════════════════════════════════════════════════════════════════

def load_wikipedia(limit: int) -> list:
    """Load your existing wikipedia_raw.json and aura_dataset.jsonl"""
    examples = []

    # Load existing formatted dataset
    if os.path.exists(EXISTING_PATH):
        print(f"\n📚 Loading existing AURA dataset ({EXISTING_PATH})...")
        with open(EXISTING_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ex = json.loads(line)
                        ex['source'] = 'wikipedia'
                        examples.append(ex)
                    except Exception:
                        continue
        print(f"  ✅ {len(examples)} examples")

    # Also load raw wikipedia and add as QA pairs
    if os.path.exists(WIKI_RAW_PATH) and len(examples) < limit:
        print(f"  Loading raw wikipedia topics...")
        with open(WIKI_RAW_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for item in raw[:limit]:
            topic   = item.get('topic', '')
            content = clean_text(item.get('content', ''))
            if not topic or not content or len(content.split()) < 30:
                continue
            examples.append(make_example(
                instruction_key = "summary",
                input_text      = f"Topic: {topic}\n\nContent: {content[:800]}",
                output_text     = content[:400],
                topic           = topic,
                source          = "wikipedia_raw",
            ))

    return examples[:limit]


# ═══════════════════════════════════════════════════════════════════
# SOURCE 5 — YOUR ANNOTATED DATA (highest quality)
# ═══════════════════════════════════════════════════════════════════

def load_annotated() -> list:
    """Load your human-annotated examples — highest quality, keep all."""
    if not os.path.exists(ANNOTATED_PATH):
        return []

    print(f"\n📚 Loading annotated dataset ({ANNOTATED_PATH})...")
    examples = []
    with open(ANNOTATED_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ex = json.loads(line)
                    ex['source']    = 'annotated'
                    ex['annotated'] = True
                    examples.append(ex)
                except Exception:
                    continue

    print(f"  ✅ {len(examples)} annotated examples (highest quality)")
    return examples


# ═══════════════════════════════════════════════════════════════════
# MERGE + DEDUPLICATE + SHUFFLE
# ═══════════════════════════════════════════════════════════════════

def deduplicate(examples: list) -> list:
    """Remove duplicate inputs."""
    seen    = set()
    cleaned = []
    for ex in examples:
        key = ex['input'][:100].strip().lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(ex)
    return cleaned


def build_final_dataset():
    os.makedirs("data", exist_ok=True)

    print("=" * 60)
    print("  AURA Unified Dataset Builder")
    print("=" * 60)

    all_examples = []
    stats        = {}

    # ── Load all sources ──────────────────────────────────────────

    # Annotated first — highest quality
    annotated = load_annotated()
    all_examples.extend(annotated)
    stats['annotated'] = len(annotated)

    # Wikipedia
    wiki = load_wikipedia(LIMITS['wikipedia'])
    all_examples.extend(wiki)
    stats['wikipedia'] = len(wiki)

    # ELI5
    eli5 = load_eli5(LIMITS['eli5'])
    all_examples.extend(eli5)
    stats['eli5'] = len(eli5)

    # Dolly
    dolly = load_dolly(LIMITS['dolly'])
    all_examples.extend(dolly)
    stats['dolly'] = len(dolly)

    # FLAN
    flan = load_flan(LIMITS['flan'])
    all_examples.extend(flan)
    stats['flan'] = len(flan)

    # ── Deduplicate ───────────────────────────────────────────────
    print(f"\n  Before dedup : {len(all_examples)} examples")
    all_examples = deduplicate(all_examples)
    print(f"  After dedup  : {len(all_examples)} examples")

    # ── Shuffle — but keep annotated examples weighted ────────────
    # Duplicate annotated examples 3x so model sees them more
    annotated_examples = [e for e in all_examples if e.get('annotated')]
    rest               = [e for e in all_examples if not e.get('annotated')]

    final = annotated_examples * 3 + rest
    random.seed(42)
    random.shuffle(final)

    # ── Save ──────────────────────────────────────────────────────
    print(f"\n  Saving {len(final)} training examples...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for ex in final:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # ── Stats ──────────────────────────────────────────────────────
    stats['total']     = len(final)
    stats['annotated_weighted'] = len(annotated_examples) * 3

    task_counts = {}
    for ex in final:
        t = ex.get('task', 'unknown')
        task_counts[t] = task_counts.get(t, 0) + 1
    stats['by_task'] = task_counts

    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Dataset Build Complete")
    print("=" * 60)
    print(f"  Annotated (×3 weighted) : {len(annotated_examples) * 3}")
    print(f"  Wikipedia               : {stats['wikipedia']}")
    print(f"  ELI5                    : {stats['eli5']}")
    print(f"  Dolly                   : {stats['dolly']}")
    print(f"  FLAN                    : {stats['flan']}")
    print(f"  ─────────────────────────────")
    print(f"  TOTAL                   : {len(final)}")
    print(f"\n  Saved to: {OUTPUT_PATH}")
    print(f"  Stats  : {STATS_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    build_final_dataset()