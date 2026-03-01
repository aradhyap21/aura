"""
AURA Dataset Cleaner & Formatter
Takes raw scraped JSON from OpenStax and Wikipedia.
Cleans, filters, and formats into training-ready JSONL.

Each output example has:
  - instruction : what to do
  - input       : the raw topic content
  - output      : the target (explanation / bullets / analogy)

This is the standard Alpaca format used for fine-tuning LLMs.

Run AFTER both scrapers:
    python formatter.py

Output:
    data/aura_dataset.jsonl   ← ready for fine-tuning
    data/aura_dataset.json    ← human-readable version
    data/stats.json           ← dataset statistics
"""

import json
import re
import os
from tqdm import tqdm

os.makedirs("data", exist_ok=True)

# ── CLEANING ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove noise from scraped text."""
    # Remove citation markers like [1], [2], [citation needed]
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove very short lines (likely navigation/UI text)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 20]
    text  = '\n'.join(lines)

    return text.strip()


def is_valid(example: dict) -> bool:
    """Check if an example is worth keeping."""
    content = example.get("content", "")
    topic   = example.get("topic", "")

    # Must have enough content
    if len(content.split()) < 50:
        return False

    # Must have a real topic title
    if len(topic.strip()) < 3:
        return False

    # Skip navigation/index pages
    skip_keywords = ["references", "bibliography", "index", "contents",
                     "further reading", "see also", "external links", "notes"]
    if any(kw in topic.lower() for kw in skip_keywords):
        return False

    return True


# ── INSTRUCTION TEMPLATES ─────────────────────────────────────────
# We create 3 different task types per example
# This teaches the model to do 3 different things

def make_explanation(topic: str, content: str) -> dict:
    """Task 1 — Simplified explanation."""
    return {
        "instruction": (
            "You are an expert tutor helping a student prepare for an exam. "
            "Read the following topic content and write a clear, simple explanation "
            "in 3-4 sentences. Avoid jargon. Make it easy to understand."
        ),
        "input"  : f"Topic: {topic}\n\nContent: {content[:800]}",
        "output" : f"[Simplified explanation of {topic} — to be filled by LLM during synthetic generation step]",
        "task"   : "explanation",
        "topic"  : topic,
        "source" : "openstax_wikipedia",
    }


def make_bullets(topic: str, content: str) -> dict:
    """Task 2 — Exam bullet points."""
    return {
        "instruction": (
            "You are an expert tutor helping a student prepare for an exam. "
            "Read the following topic and extract exactly 5 key exam-ready bullet points. "
            "Each bullet should be a fact or concept the student must remember. "
            "Be concise and specific."
        ),
        "input"  : f"Topic: {topic}\n\nContent: {content[:800]}",
        "output" : f"[5 exam bullet points for {topic} — to be filled by LLM during synthetic generation step]",
        "task"   : "bullets",
        "topic"  : topic,
        "source" : "openstax_wikipedia",
    }


def make_analogy(topic: str, content: str) -> dict:
    """Task 3 — Real-world analogy."""
    return {
        "instruction": (
            "You are an expert tutor. "
            "Create a simple, memorable real-world analogy that explains the following topic. "
            "The analogy should make the concept immediately clear to a student "
            "who has never studied this before."
        ),
        "input"  : f"Topic: {topic}\n\nContent: {content[:600]}",
        "output" : f"[Real-world analogy for {topic} — to be filled by LLM during synthetic generation step]",
        "task"   : "analogy",
        "topic"  : topic,
        "source" : "openstax_wikipedia",
    }


# ── LOAD RAW DATA ─────────────────────────────────────────────────

def load_raw() -> list[dict]:
    all_raw = []

    for fname in ["data/openstax_raw.json", "data/wikipedia_raw.json"]:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"  Loaded {len(data)} examples from {fname}")
            all_raw.extend(data)
        else:
            print(f"  ⚠ {fname} not found — run the scrapers first")

    return all_raw


# ── MAIN ──────────────────────────────────────────────────────────

def format_dataset():
    print("=" * 60)
    print("AURA Dataset Formatter")
    print("=" * 60)

    # Load
    print("\n📂 Loading raw data...")
    raw = load_raw()
    print(f"   Total raw examples: {len(raw)}")

    # Clean and filter
    print("\n🧹 Cleaning and filtering...")
    cleaned = []
    for ex in tqdm(raw):
        ex["content"] = clean_text(ex.get("content", ""))
        ex["topic"]   = ex.get("topic", "").strip()
        if is_valid(ex):
            cleaned.append(ex)

    print(f"   After filtering: {len(cleaned)} examples kept")

    # Remove duplicates by topic name
    seen   = set()
    unique = []
    for ex in cleaned:
        key = ex["topic"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    print(f"   After deduplication: {len(unique)} unique topics")

    # Generate 3 training examples per topic
    print("\n⚙️  Generating training examples (3 per topic)...")
    training = []
    for ex in tqdm(unique):
        topic   = ex["topic"]
        content = ex["content"]

        training.append(make_explanation(topic, content))
        training.append(make_bullets(topic, content))
        training.append(make_analogy(topic, content))

    print(f"   Total training examples: {len(training)}")

    # Save as JSONL (one JSON object per line — standard for fine-tuning)
    jsonl_path = "data/aura_dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in training:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save as readable JSON too
    json_path = "data/aura_dataset.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training, f, indent=2, ensure_ascii=False)

    # Stats
    stats = {
        "total_raw"       : len(raw),
        "after_filtering" : len(cleaned),
        "unique_topics"   : len(unique),
        "total_training"  : len(training),
        "task_breakdown"  : {
            "explanation" : len([t for t in training if t["task"] == "explanation"]),
            "bullets"     : len([t for t in training if t["task"] == "bullets"]),
            "analogy"     : len([t for t in training if t["task"] == "analogy"]),
        },
        "sources": {
            "openstax"  : len([t for t in unique if t.get("source") == "openstax"]),
            "wikipedia" : len([t for t in unique if t.get("source") == "wikipedia"]),
        }
    }

    with open("data/stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Dataset ready!")
    print(f"   JSONL → {jsonl_path}")
    print(f"   JSON  → {json_path}")
    print(f"   Stats → data/stats.json")
    print(f"\n   Total training examples: {len(training)}")
    print(f"   Explanation tasks : {stats['task_breakdown']['explanation']}")
    print(f"   Bullet tasks      : {stats['task_breakdown']['bullets']}")
    print(f"   Analogy tasks     : {stats['task_breakdown']['analogy']}")
    print("=" * 60)


if __name__ == "__main__":
    format_dataset()