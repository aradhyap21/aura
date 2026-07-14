"""
AURA Dataset Scraper — Wikipedia
Scrapes educational topic articles from Wikipedia.
Extracts: article sections as topic-content pairs.
Uses the official wikipedia-api library (clean, no HTML parsing needed).

Run:
    python wikipedia_scraper.py

Output:
    data/wikipedia_raw.json
"""

import wikipediaapi
import json
import time
import os

os.makedirs("data", exist_ok=True)

# ── Wikipedia API setup ───────────────────────────────────────────
wiki = wikipediaapi.Wikipedia(
    language   = "en",
    user_agent = "AURA-DataCollector/1.0 (educational research project)"
)

# ── Topics to scrape (educational, exam-relevant) ─────────────────
TOPICS = [
    # Computer Science
    "Machine learning", "Deep learning", "Neural network",
    "Artificial intelligence", "Natural language processing",
    "Data structure", "Algorithm", "Operating system",
    "Computer network", "Database", "Sorting algorithm",
    "Object-oriented programming", "Software engineering",
    "Cybersecurity", "Cloud computing", "Big data",

    # Mathematics
    "Calculus", "Linear algebra", "Probability", "Statistics",
    "Differential equation", "Number theory", "Set theory",
    "Matrix (mathematics)", "Graph theory", "Trigonometry",

    # Physics
    "Quantum mechanics", "Thermodynamics", "Electromagnetism",
    "Classical mechanics", "Optics", "Relativity",
    "Wave", "Electric current", "Magnetism",

    # Chemistry
    "Chemical bond", "Periodic table", "Acid-base reaction",
    "Organic chemistry", "Thermochemistry", "Electrochemistry",

    # Biology
    "Cell (biology)", "DNA", "Protein", "Evolution",
    "Genetics", "Photosynthesis", "Ecosystem", "Human anatomy",

    # Economics
    "Supply and demand", "Inflation", "GDP", "Market structure",
    "Microeconomics", "Macroeconomics", "Fiscal policy",

    # Psychology
    "Cognitive psychology", "Memory", "Learning", "Motivation",
    "Perception", "Developmental psychology", "Behaviorism",
]


def extract_sections(page, topic_name: str, depth: int = 0) -> list[dict]:
    """
    Recursively extract all sections of a Wikipedia article.
    Each section = one training example.
    """
    results = []

    # Skip very short sections
    if len(page.text.split()) < 20:
        return results

    # Add this section
    results.append({
        "source"  : "wikipedia",
        "topic"   : page.title if depth == 0 else f"{topic_name} — {page.title}",
        "content" : page.text.strip(),
        "url"     : f"https://en.wikipedia.org/wiki/{topic_name.replace(' ', '_')}",
    })

    # Recurse into subsections (max depth 2 to avoid going too deep)
    if depth < 2:
        for section in page.sections:
            results.extend(extract_sections(section, topic_name, depth + 1))

    return results


def scrape_wikipedia() -> list[dict]:
    all_data = []
    failed   = []

    print(f"Scraping {len(TOPICS)} topics from Wikipedia...\n")

    for i, topic in enumerate(TOPICS):
        print(f"[{i+1}/{len(TOPICS)}] {topic}", end=" ... ")

        page = wiki.page(topic)

        if not page.exists():
            print("✗ not found")
            failed.append(topic)
            continue

        examples = extract_sections(page, topic)

        # Filter: keep only sections with enough words
        examples = [e for e in examples if len(e["content"].split()) >= 40]

        all_data.extend(examples)
        print(f"✅ {len(examples)} sections")

        time.sleep(0.5)  # polite delay

    print(f"\n✅ Wikipedia done — {len(all_data)} examples collected")
    if failed:
        print(f"⚠ Failed topics: {failed}")

    return all_data


if __name__ == "__main__":
    print("=" * 60)
    print("AURA Dataset Scraper — Wikipedia")
    print("=" * 60)

    data = scrape_wikipedia()

    output_path = "data/wikipedia_raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(data)} examples → {output_path}")