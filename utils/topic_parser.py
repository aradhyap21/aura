"""
AURA v3 — Topic Parser
Complete rewrite. Much more aggressive heading detection.
Works on messy real-world PDFs, DOCX, and PPTX.

Strategy:
- 5 independent detection methods run on every line
- Any line passing ANY method = heading candidate
- Topics vs subtopics separated by level
- ZERO caps on output — shows every topic and subtopic found
"""

import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from models.schemas import Topic, Subtopic, Priority

# ── NLTK setup ────────────────────────────────────────────────────
for resource, name in [
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(name, quiet=True)

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update({
    "also", "may", "use", "used", "using", "one", "two", "three",
    "first", "second", "third", "example", "figure", "table",
    "chapter", "section", "page", "note", "following", "shown",
    "refer", "thus", "hence", "however", "therefore", "whereas",
    "slide", "introduction", "conclusion", "summary", "overview",
    "definition", "versus", "within", "between", "through", "where",
    "which", "that", "this", "from", "with", "have", "been", "will",
})


# ─────────────────────────────────────────────────────────────────
# 5 INDEPENDENT HEADING DETECTION METHODS
# Each returns (title_string | None, level_int)
# level 1 = main topic,  level 2+ = subtopic,  0 = not a heading
# ─────────────────────────────────────────────────────────────────

def method_markdown(line: str):
    m = re.match(r'^(#{1,4})\s+(.+)$', line)
    if m:
        level = len(m.group(1))
        title = m.group(2).strip()
        if len(title) >= 2:
            return title, level
    return None, 0


def method_numbered(line: str):
    # 1.1.1 pattern → level 3
    m = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)$', line)
    if m and len(m.group(2).strip()) >= 2:
        return m.group(2).strip(), 3

    # 1.1 pattern → level 2
    m = re.match(r'^(\d+\.\d+)\s+(.+)$', line)
    if m and len(m.group(2).strip()) >= 2:
        return m.group(2).strip(), 2

    # 1. Title → level 1
    m = re.match(r'^(\d{1,2})[\.]\s+([A-Z].{2,60})$', line)
    if m:
        return m.group(2).strip(), 1

    # A. Title → level 1
    m = re.match(r'^([A-Z])[\.]\s+(.{3,60})$', line)
    if m:
        return m.group(2).strip(), 1

    # a) or b) → level 2
    m = re.match(r'^([a-z])\)\s+(.{3,60})$', line)
    if m:
        return m.group(2).strip(), 2

    # Roman numerals I. II. → level 2
    m = re.match(r'^([IVXivx]+)[\.]\s+(.{3,60})$', line)
    if m:
        return m.group(2).strip(), 2

    return None, 0


def method_keyword_prefix(line: str):
    patterns = [
        (r'^(Chapter\s+\d+[\s\:\-\.]*.*)',    1),
        (r'^(Unit\s+\d+[\s\:\-\.]*.*)',        1),
        (r'^(Module\s+\d+[\s\:\-\.]*.*)',      1),
        (r'^(Lesson\s+\d+[\s\:\-\.]*.*)',      1),
        (r'^(Part\s+[IVX\d]+[\s\:\-\.]*.*)',   1),
        (r'^(Section\s+\d+[\s\:\-\.]*.*)',     1),
        (r'^(Lab\s+\d+[\s\:\-\.]*.*)',         1),
        (r'^(Lecture\s+\d+[\s\:\-\.]*.*)',     1),
        (r'^(Week\s+\d+[\s\:\-\.]*.*)',        1),
        (r'^(Introduction\s+to\s+.+)',         1),
        (r'^(Overview\s+of\s+.+)',             1),
        (r'^(Types\s+of\s+.+)',                2),
        (r'^(Topic\s*[\:\-]\s*.+)',            2),
        (r'^(Subtopic\s*[\:\-]\s*.+)',         2),
        (r'^(Concept\s*[\:\-]\s*.+)',          2),
        (r'^(Definition\s*[\:\-]\s*.+)',       2),
    ]
    for pattern, level in patterns:
        m = re.match(pattern, line, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            if len(title) >= 3:
                return title, level
    return None, 0


def method_caps(line: str):
    if len(line) < 4 or len(line) > 100:
        return None, 0
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return None, 0
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio < 0.85:
        return None, 0
    words = line.split()
    if len(words) < 2:
        return None, 0
    return line.title(), 1


def method_title_case(line: str):
    if len(line) < 5 or len(line) > 100:
        return None, 0
    # Reject sentences (ends with punctuation)
    if line[-1] in '.!?,;:':
        return None, 0
    words = line.split()
    if len(words) < 2 or len(words) > 10:
        return None, 0
    skip = {'a','an','the','of','in','on','at','to','for','and','or',
            'but','with','by','from','as','is','are','was','were'}
    content_words = [w for w in words if w.lower() not in skip]
    if not content_words:
        return None, 0
    cap_words = [w for w in content_words if w and w[0].isupper()]
    cap_ratio = len(cap_words) / len(content_words)
    if cap_ratio >= 0.75:
        return line, 1
    return None, 0


def classify_line(line: str):
    """
    Run all 5 methods. First match wins.
    Returns (title | None, level)
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return None, 0

    for method in [method_markdown, method_numbered, method_keyword_prefix,
                   method_caps, method_title_case]:
        title, level = method(stripped)
        if title:
            return title, level

    return None, 0


# ─────────────────────────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 8) -> list:
    words    = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', text.lower())
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 3]
    if not filtered:
        return []
    if len(filtered) < 15:
        return [w for w, _ in Counter(filtered).most_common(top_n)]
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 15]
    if len(sentences) < 2:
        return [w for w, _ in Counter(filtered).most_common(top_n)]
    try:
        vec    = TfidfVectorizer(
            max_features  = 200,
            ngram_range   = (1, 2),
            stop_words    = 'english',
            token_pattern = r'\b[a-zA-Z][a-zA-Z\-]{2,}\b',
        )
        matrix = vec.fit_transform(sentences)
        names  = vec.get_feature_names_out()
        scores = matrix.mean(axis=0).A1
        ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [w for w, _ in ranked[:top_n]]
    except Exception:
        return [w for w, _ in Counter(filtered).most_common(top_n)]


# ─────────────────────────────────────────────────────────────────
# STRUCTURE PARSER
# ─────────────────────────────────────────────────────────────────

def parse_structure(raw_text: str) -> list:
    lines = raw_text.splitlines()

    topics           = []
    current_topic    = None
    current_subtopic = None

    def close_subtopic():
        nonlocal current_subtopic
        if current_subtopic is not None and current_topic is not None:
            content = ' '.join(current_subtopic['lines']).strip()
            if len(content.split()) >= 3:
                current_topic['subtopics'].append({
                    'title'  : current_subtopic['title'],
                    'content': content,
                })
        current_subtopic = None

    def close_topic():
        nonlocal current_topic
        close_subtopic()
        if current_topic is not None:
            content = ' '.join(current_topic['lines']).strip()
            if len(content.split()) >= 3 or len(current_topic['subtopics']) > 0:
                topics.append({
                    'title'    : current_topic['title'],
                    'content'  : content,
                    'subtopics': current_topic['subtopics'],
                    'line_idx' : current_topic['line_idx'],
                })
        current_topic = None

    for i, raw_line in enumerate(lines):
        line = re.sub(r'[^\x20-\x7E]', ' ', raw_line)
        line = re.sub(r'[ \t]+', ' ', line).strip()
        if not line:
            continue

        title, level = classify_line(line)

        if title and level == 1:
            close_topic()
            current_topic = {
                'title'    : title,
                'lines'    : [],
                'subtopics': [],
                'line_idx' : i,
            }
            current_subtopic = None

        elif title and level >= 2:
            if current_topic is not None:
                close_subtopic()
                current_subtopic = {'title': title, 'lines': []}
            else:
                # No parent yet — promote to topic
                close_topic()
                current_topic = {
                    'title'    : title,
                    'lines'    : [],
                    'subtopics': [],
                    'line_idx' : i,
                }

        else:
            if current_subtopic is not None:
                current_subtopic['lines'].append(line)
            if current_topic is not None:
                current_topic['lines'].append(line)

    close_topic()
    return topics


def parse_by_paragraphs(raw_text: str) -> list:
    """Fallback: no headings found — split by blank lines."""
    blocks = re.split(r'\n{2,}', raw_text)
    topics = []
    for i, block in enumerate(blocks):
        block = block.strip()
        if len(block.split()) < 8:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', block)
        title = sentences[0][:80].strip() if sentences else f"Section {i+1}"
        topics.append({
            'title'    : title or f"Section {i+1}",
            'content'  : block,
            'subtopics': [],
            'line_idx' : i,
        })
    return topics


# ─────────────────────────────────────────────────────────────────
# SCORING & PRIORITY
# ─────────────────────────────────────────────────────────────────

def score_topic(content: str, position: float, total_words: int) -> float:
    words      = content.split()
    n          = len(words)
    size_score = min(n / max(total_words * 0.03, 1), 1.0)
    pos_score  = 1.0 - (0.4 * position)
    meaningful = [w for w in words if w.lower() not in STOPWORDS and len(w) > 3]
    density    = len(meaningful) / max(n, 1)
    return round(0.45 * size_score + 0.30 * pos_score + 0.25 * density, 4)


def assign_priority(score: float, all_scores: list) -> Priority:
    if not all_scores:
        return Priority.LOW
    sorted_s = sorted(all_scores, reverse=True)
    n        = len(sorted_s)
    high_cut = sorted_s[max(0, int(n * 0.20) - 1)]
    med_cut  = sorted_s[max(0, int(n * 0.55) - 1)]
    if score >= high_cut:
        return Priority.HIGH
    elif score >= med_cut:
        return Priority.MEDIUM
    return Priority.LOW


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def parse_topics(raw_text: str) -> list:
    """
    Returns ALL topics and ALL subtopics. No list size limits.
    Sorted: High priority first, then Medium, then Low.
    """
    raw_topics = parse_structure(raw_text)

    if len(raw_topics) < 2:
        raw_topics = parse_by_paragraphs(raw_text)

    if not raw_topics:
        return []

    total_words = len(raw_text.split())
    total_lines = max(len(raw_text.splitlines()), 1)

    scored = []
    for t in raw_topics:
        position = t['line_idx'] / total_lines
        s        = score_topic(t['content'], position, total_words)
        scored.append((t, s))

    all_scores = [s for _, s in scored]
    result     = []

    for raw_t, score in scored:
        priority = assign_priority(score, all_scores)
        keywords = extract_keywords(raw_t['content'], top_n=8)

        subtopic_objects = []
        for raw_sub in raw_t['subtopics']:
            sub_kw = extract_keywords(raw_sub['content'], top_n=5)
            subtopic_objects.append(Subtopic(
                title      = raw_sub['title'],
                content    = raw_sub['content'][:600],
                word_count = len(raw_sub['content'].split()),
                keywords   = sub_kw,
            ))

        result.append(Topic(
            title      = raw_t['title'],
            priority   = priority,
            score      = score,
            word_count = len(raw_t['content'].split()),
            keywords   = keywords,
            subtopics  = subtopic_objects,
        ))

    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    result.sort(key=lambda t: (order[t.priority], -t.score))
    return result