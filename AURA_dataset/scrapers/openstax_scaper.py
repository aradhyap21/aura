"""
AURA Dataset Scraper — OpenStax (Fixed)
Uses archive.cnx.org JSON API — much more reliable than HTML scraping.

Run:
    python scrapers/openstax_scraper.py

Output:
    data/openstax_raw.json
"""

import requests
import json
import time
import os
import re
from bs4 import BeautifulSoup

os.makedirs("data", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AURA-DataCollector/1.0)"
}

# ── OpenStax Book UUIDs from archive.cnx.org ─────────────────────
# Format: (book_name, uuid)
BOOKS = [
    ("Psychology 2e",               "4664c267-cd62-4a99-8b28-1cb9b3aee347"),
    ("Biology 2e",                  "8d50a0af-948b-4204-a71d-4826cba765b8"),
    ("Principles of Economics 3e",  "bc498e1f-efe9-43a0-8dea-d3569ad09a82"),
    ("College Physics 2e",          "e54516ad-23d2-421b-9b8f-9e5026a92d37"),
    ("US History",                  "a7ba2fb8-8925-4987-b182-5f4429d48daa"),
    ("Chemistry 2e",                "85abf193-2bd2-4908-8563-90b8a7ac8df6"),
    ("Introduction to Sociology 3e","02040312-72c8-441e-a685-20e9333f3e1d"),
    ("Anatomy and Physiology 2e",   "40a35dc4-5703-4994-8a64-7f9da1c8d42f"),
    ("Calculus Volume 1",           "13ac107a-f15f-49d2-97e8-60ab2e3b519c"),
    ("University Physics Volume 1", "d50f6e32-0fda-46ef-a362-9bd36ca7c97d"),
]

BASE_URL = "https://archive.cnx.org/contents"


def clean_html(html_text: str) -> str:
    """Strip HTML tags and clean up text."""
    if not html_text:
        return ""
    soup = BeautifulSoup(html_text, "html.parser")
    # Remove figures, footnotes, math
    for tag in soup.find_all(["figure", "math", "sup", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def get_book_tree(uuid: str) -> dict | None:
    """Fetch the full book table of contents."""
    url = f"{BASE_URL}/{uuid}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            return r.json()
        print(f"  ⚠ Status {r.status_code}")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def get_page_content(page_id: str) -> str:
    """Fetch a single page's HTML content."""
    url = f"{BASE_URL}/{page_id}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return data.get("content", "")
        return ""
    except Exception:
        return ""


def walk_tree(tree: list, book_name: str, results: list, depth: int = 0):
    """
    Recursively walk the book tree.
    Each leaf node = a page = a topic.
    """
    for node in tree:
        title    = node.get("title", "").strip()
        contents = node.get("contents", [])
        node_id  = node.get("id", "")

        if contents:
            # This is a chapter — recurse into it
            walk_tree(contents, book_name, results, depth + 1)
        elif node_id and title:
            # This is a leaf page — fetch content
            html = get_page_content(node_id)
            if html:
                text = clean_html(html)
                word_count = len(text.split())
                if word_count >= 80:
                    results.append({
                        "source"    : "openstax",
                        "book"      : book_name,
                        "topic"     : title,
                        "content"   : text,
                        "url"       : f"https://openstax.org/books/{book_name.lower().replace(' ', '-')}/pages/{node_id}",
                    })
            time.sleep(0.8)  # polite delay


def scrape_openstax() -> list[dict]:
    all_data = []

    for book_name, uuid in BOOKS:
        print(f"\n📚 {book_name}")

        tree_data = get_book_tree(uuid)
        if not tree_data:
            print(f"  ✗ Could not fetch book tree")
            continue

        # Book tree is in tree_data["tree"]["contents"]
        book_tree = tree_data.get("tree", {}).get("contents", [])
        if not book_tree:
            print(f"  ✗ Empty book tree")
            continue

        book_results = []
        walk_tree(book_tree, book_name, book_results)
        all_data.extend(book_results)
        print(f"  ✅ {len(book_results)} sections scraped")

    return all_data


if __name__ == "__main__":
    print("=" * 60)
    print("AURA Dataset Scraper — OpenStax (via archive.cnx.org API)")
    print("=" * 60)

    data = scrape_openstax()

    output_path = "data/openstax_raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Saved {len(data)} examples → {output_path}")