"""
AURA Dataset Scraper — OpenStax
Scrapes free textbook content from OpenStax.
Extracts: chapter titles, section headings, body content.
Saves as structured JSON ready for fine-tuning.

Run:
    python openstax_scraper.py

Output:
    data/openstax_raw.json
"""

import requests
import json
import time
import os
from bs4 import BeautifulSoup

# ── Output folder ─────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AURA-DataCollector/1.0)"
}

# ── OpenStax Books to scrape ──────────────────────────────────────
# Each entry: (book_name, book_slug_url)
OPENSTAX_BOOKS = [
    ("Introduction to Sociology",       "https://openstax.org/books/introduction-sociology-3e/pages/1-introduction"),
    ("Psychology 2e",                   "https://openstax.org/books/psychology-2e/pages/1-introduction"),
    ("Biology 2e",                      "https://openstax.org/books/biology-2e/pages/1-introduction"),
    ("Principles of Economics 3e",      "https://openstax.org/books/principles-economics-3e/pages/1-introduction"),
    ("College Physics",                 "https://openstax.org/books/college-physics-2e/pages/1-introduction"),
    ("Introduction to Statistics",      "https://openstax.org/books/introductory-statistics/pages/1-introduction"),
    ("Calculus Volume 1",               "https://openstax.org/books/calculus-volume-1/pages/1-introduction"),
    ("Anatomy and Physiology",          "https://openstax.org/books/anatomy-and-physiology-2e/pages/1-introduction"),
    ("US History",                      "https://openstax.org/books/us-history/pages/1-introduction"),
    ("Chemistry 2e",                    "https://openstax.org/books/chemistry-2e/pages/1-introduction"),
]


def get_page(url: str) -> BeautifulSoup | None:
    """Fetch a page and return BeautifulSoup object."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
        else:
            print(f"  ⚠ Status {r.status_code} for {url}")
            return None
    except Exception as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return None


def get_all_chapter_links(book_url: str) -> list[str]:
    """
    Get all chapter/section page links from a book's table of contents.
    OpenStax loads TOC in the sidebar — we extract all page links.
    """
    soup = get_page(book_url)
    if not soup:
        return []

    links = []
    base = "https://openstax.org"

    # OpenStax TOC links are in <nav> or sidebar with class containing 'toc'
    toc = soup.find("nav", {"aria-label": True}) or soup.find("ol", class_=lambda c: c and "toc" in c.lower())

    if toc:
        for a in toc.find_all("a", href=True):
            href = a["href"]
            if "/pages/" in href:
                full = base + href if href.startswith("/") else href
                if full not in links:
                    links.append(full)
    else:
        # Fallback: get all /pages/ links on the page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/pages/" in href and "introduction" not in href.lower():
                full = base + href if href.startswith("/") else href
                if full not in links:
                    links.append(full)

    return links[:50]  # cap at 50 pages per book


def extract_page_content(url: str, book_name: str) -> list[dict]:
    """
    Extract topic-content pairs from a single OpenStax page.
    Each section heading + its paragraphs = one training example.
    """
    soup = get_page(url)
    if not soup:
        return []

    results = []

    # Main content area
    main = (
        soup.find("div", class_=lambda c: c and "content" in c.lower()) or
        soup.find("main") or
        soup.find("article")
    )
    if not main:
        return []

    # Get page title (chapter/section name)
    page_title = ""
    h1 = main.find("h1") or soup.find("h1")
    if h1:
        page_title = h1.get_text(strip=True)

    # Walk through headings and collect content under each
    current_heading = page_title
    current_paragraphs = []

    for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        tag = elem.name

        if tag in ["h1", "h2", "h3", "h4"]:
            # Save previous section
            if current_heading and current_paragraphs:
                content = " ".join(current_paragraphs).strip()
                if len(content.split()) >= 30:  # only if enough content
                    results.append({
                        "source"    : "openstax",
                        "book"      : book_name,
                        "url"       : url,
                        "topic"     : current_heading,
                        "content"   : content,
                    })
            current_heading = elem.get_text(strip=True)
            current_paragraphs = []

        elif tag in ["p", "li"]:
            text = elem.get_text(strip=True)
            if len(text) > 30:  # skip very short lines
                current_paragraphs.append(text)

    # Save last section
    if current_heading and current_paragraphs:
        content = " ".join(current_paragraphs).strip()
        if len(content.split()) >= 30:
            results.append({
                "source" : "openstax",
                "book"   : book_name,
                "url"    : url,
                "topic"  : current_heading,
                "content": content,
            })

    return results


def scrape_openstax() -> list[dict]:
    """Main scraper — loops over all books and pages."""
    all_data = []

    for book_name, start_url in OPENSTAX_BOOKS:
        print(f"\n📚 Scraping: {book_name}")

        chapter_links = get_all_chapter_links(start_url)
        print(f"   Found {len(chapter_links)} pages")

        for i, url in enumerate(chapter_links):
            print(f"   [{i+1}/{len(chapter_links)}] {url.split('/')[-1]}", end="\r")
            examples = extract_page_content(url, book_name)
            all_data.extend(examples)
            time.sleep(1.2)  # polite delay — don't hammer the server

        print(f"   ✅ {book_name} done — {len(all_data)} total examples so far")

    return all_data


if __name__ == "__main__":
    print("=" * 60)
    print("AURA Dataset Scraper — OpenStax")
    print("=" * 60)

    data = scrape_openstax()

    # Save raw
    output_path = "data/openstax_raw.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done! Saved {len(data)} examples → {output_path}")