"""
AURA v3 — Semantic Topic Extractor
Replaces topic_parser.py completely.

How it works (same as how ChatGPT/NotebookLM does it):

1. Split document into sentences
2. Convert every sentence into a meaning vector (embedding)
   using a pre-trained Sentence Transformer model
3. Reduce dimensions with UMAP
4. Cluster sentences by meaning using HDBSCAN
   → Each cluster = one topic (works even with NO headings)
5. Extract best title for each cluster using TF-IDF keywords
6. Detect subtopics within each cluster (sub-clustering)
7. Score and rank topics by information density + uniqueness
8. Assign High / Medium / Low priority

Works on:
- PDFs with headings
- PDFs with NO headings (lecture notes)
- PPTX slides
- DOCX files
- Scanned PDFs (after OCR in extractor.py)
"""

import re
import numpy as np
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
from nltk.tokenize import sent_tokenize

STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update({
    "also", "may", "use", "used", "using", "one", "two", "three",
    "first", "second", "third", "example", "figure", "table",
    "chapter", "section", "page", "note", "following", "shown",
    "refer", "thus", "hence", "however", "therefore", "whereas",
    "slide", "introduction", "conclusion", "summary", "overview",
    "definition", "versus", "within", "between", "through",
    "which", "that", "this", "from", "with", "have", "been", "will",
    "said", "say", "also", "well", "just", "can", "get", "make",
})

# ── Lazy-load heavy models (only when needed) ─────────────────────
_embedder = None

def get_embedder():
    """Load sentence transformer model once and cache it."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            # all-MiniLM-L6-v2: fast, small (80MB), excellent quality
            # Downloads automatically on first run
            print("  Loading embedding model (first run — downloads ~80MB)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Embedding model loaded.")
        except ImportError:
            print("  ⚠ sentence-transformers not installed. Falling back to TF-IDF mode.")
            _embedder = "tfidf"
    return _embedder


# ─────────────────────────────────────────────────────────────────
# STEP 1 — SENTENCE SPLITTING
# ─────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split document text into clean sentences.
    Handles bullet points, numbered lists, and paragraph breaks.
    """
    # Clean text first
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)

    sentences = []

    # Split by newlines first (handles bullet/numbered lists)
    paragraphs = text.split('\n')

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Use NLTK sentence tokenizer on each paragraph
        try:
            sents = sent_tokenize(para)
        except Exception:
            sents = [para]

        for s in sents:
            s = s.strip()
            # Keep sentences with at least 6 words
            if len(s.split()) >= 6:
                sentences.append(s)

    return sentences


# ─────────────────────────────────────────────────────────────────
# STEP 2 — EMBED SENTENCES
# ─────────────────────────────────────────────────────────────────

def embed_sentences(sentences: list[str]) -> np.ndarray:
    """
    Convert sentences to meaning vectors.
    Uses Sentence Transformers if available, TF-IDF fallback otherwise.
    """
    embedder = get_embedder()

    if embedder == "tfidf":
        # Fallback: TF-IDF vectors
        vec = TfidfVectorizer(max_features=500, stop_words='english')
        try:
            return vec.fit_transform(sentences).toarray()
        except Exception:
            # Last resort: random vectors (won't cluster well but won't crash)
            return np.random.rand(len(sentences), 50)

    # Sentence Transformers — proper semantic embeddings
    embeddings = embedder.encode(
        sentences,
        batch_size      = 32,
        show_progress_bar = False,
        convert_to_numpy  = True,
    )
    return embeddings


# ─────────────────────────────────────────────────────────────────
# STEP 3 — CLUSTER INTO TOPICS
# ─────────────────────────────────────────────────────────────────

def cluster_sentences(embeddings: np.ndarray, n_sentences: int) -> np.ndarray:
    """
    Cluster sentence embeddings into topic groups.

    Uses UMAP + HDBSCAN if available (best quality).
    Falls back to KMeans if not installed.

    Returns: array of cluster labels (one per sentence)
             -1 means "noise" / doesn't belong to any cluster
    """

    # ── Try UMAP + HDBSCAN (best) ─────────────────────────────────
    try:
        import umap
        import hdbscan

        # Reduce dimensions first (speeds up clustering, improves quality)
        n_components = min(10, embeddings.shape[1], n_sentences - 2)
        reducer = umap.UMAP(
            n_components  = n_components,
            n_neighbors   = max(5, min(15, n_sentences // 10)),
            min_dist      = 0.0,
            metric        = 'cosine',
            random_state  = 42,
        )
        reduced = reducer.fit_transform(embeddings)

        # Cluster — min_cluster_size controls how many topics you get
        # Larger doc = larger min_cluster_size = fewer, broader topics
        min_size = max(3, n_sentences // 20)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size  = min_size,
            min_samples       = 2,
            metric            = 'euclidean',
            cluster_selection_method = 'eom',
        )
        labels = clusterer.fit_predict(reduced)
        return labels

    except ImportError:
        pass

    # ── Fallback: KMeans (always available via sklearn) ───────────
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    # Normalize embeddings for cosine similarity
    normed = normalize(embeddings)

    # Estimate number of clusters
    n_clusters = max(3, min(20, n_sentences // 15))

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(normed)
    return labels


# ─────────────────────────────────────────────────────────────────
# STEP 4 — EXTRACT TOPIC TITLE FROM CLUSTER
# ─────────────────────────────────────────────────────────────────

def extract_topic_title(sentences: list[str], max_words: int = 6) -> str:
    """
    Given a group of sentences (one cluster/topic),
    extract the best 2-6 word title using TF-IDF.

    Strategy:
    1. Find the most distinctive keywords in this cluster
    2. Use them to form a clean title
    """
    if not sentences:
        return "Unknown Topic"

    # Get all words
    words = []
    for s in sentences:
        words.extend(re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', s.lower()))

    filtered = [w for w in words if w not in STOPWORDS and len(w) > 3]
    if not filtered:
        return "Unknown Topic"

    # Get top keywords by frequency
    freq    = Counter(filtered)
    top_kws = [w for w, _ in freq.most_common(max_words)]

    if not top_kws:
        return "Unknown Topic"

    # Try to form a natural-sounding title
    # Capitalize each word
    title = ' '.join(w.capitalize() for w in top_kws[:4])
    return title


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Extract top keywords from a text block using TF-IDF."""
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 15]
    if not sentences:
        words    = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
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
        words    = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]


# ─────────────────────────────────────────────────────────────────
# STEP 5 — DETECT SUBTOPICS WITHIN EACH CLUSTER
# ─────────────────────────────────────────────────────────────────

def detect_subtopics(
    sentences   : list[str],
    embeddings  : np.ndarray,
) -> list[dict]:
    """
    Within a topic cluster, find subtopics by sub-clustering.
    Only applied to large topics (10+ sentences).
    """
    if len(sentences) < 10:
        return []

    try:
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.preprocessing import normalize

        normed     = normalize(embeddings)
        n_sub      = max(2, min(5, len(sentences) // 5))
        sub_model  = AgglomerativeClustering(n_clusters=n_sub, metric='cosine', linkage='average')
        sub_labels = sub_model.fit_predict(normed)

        subtopics = []
        for sub_id in set(sub_labels):
            sub_sents = [sentences[i] for i, l in enumerate(sub_labels) if l == sub_id]
            if len(sub_sents) < 2:
                continue
            sub_content = ' '.join(sub_sents)
            sub_title   = extract_topic_title(sub_sents, max_words=5)
            sub_kws     = extract_keywords(sub_content, top_n=5)
            subtopics.append({
                'title'     : sub_title,
                'content'   : sub_content[:600],
                'word_count': len(sub_content.split()),
                'keywords'  : sub_kws,
            })

        return subtopics

    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────
# STEP 6 — SCORE AND RANK TOPICS
# ─────────────────────────────────────────────────────────────────

def score_topic_semantic(
    sentences        : list[str],
    embeddings       : np.ndarray,
    all_embeddings   : np.ndarray,
    position         : float,
    total_sentences  : int,
) -> float:
    """
    Score a topic using 4 signals:

    1. Size         — how many sentences / proportion of document
    2. Uniqueness   — how different this topic is from others (cosine distance)
    3. Density      — average similarity within cluster (coherence)
    4. Position     — topics appearing earlier score slightly higher
    """
    n = len(sentences)

    # Signal 1: Size (normalized)
    size_score = min(n / max(total_sentences * 0.05, 1), 1.0)

    # Signal 2: Uniqueness — centroid distance from rest of doc
    if len(embeddings) > 0 and len(all_embeddings) > len(embeddings):
        centroid     = embeddings.mean(axis=0, keepdims=True)
        all_centroid = all_embeddings.mean(axis=0, keepdims=True)
        sim          = cosine_similarity(centroid, all_centroid)[0][0]
        uniqueness   = 1.0 - float(sim)  # less similar to whole doc = more unique
    else:
        uniqueness = 0.5

    # Signal 3: Coherence — how tightly clustered the sentences are
    if len(embeddings) > 1:
        centroid  = embeddings.mean(axis=0, keepdims=True)
        sims      = cosine_similarity(embeddings, centroid).flatten()
        coherence = float(sims.mean())
    else:
        coherence = 0.5

    # Signal 4: Position bonus
    position_score = 1.0 - (0.3 * position)

    score = (
        0.35 * size_score     +
        0.30 * uniqueness     +
        0.20 * coherence      +
        0.15 * position_score
    )
    return round(float(score), 4)


def assign_priority(score: float, all_scores: list[float]) -> Priority:
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
    Full semantic pipeline. Called by analyze_service.py.

    Returns ALL topics found with ALL subtopics.
    Works even on documents with NO headings.
    """

    # ── Step 1: Split into sentences ──────────────────────────────
    sentences = split_sentences(raw_text)

    if len(sentences) < 5:
        return []

    # ── Step 2: Embed all sentences ───────────────────────────────
    print(f"  Embedding {len(sentences)} sentences...")
    all_embeddings = embed_sentences(sentences)

    # ── Step 3: Cluster into topics ───────────────────────────────
    print(f"  Clustering into topics...")
    labels = cluster_sentences(all_embeddings, len(sentences))

    # ── Step 4: Group sentences by cluster ────────────────────────
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue  # noise — skip
        if label not in clusters:
            clusters[label] = {'sentences': [], 'indices': []}
        clusters[label]['sentences'].append(sentences[i])
        clusters[label]['indices'].append(i)

    if not clusters:
        # All sentences were noise — treat whole doc as one topic
        clusters = {0: {'sentences': sentences, 'indices': list(range(len(sentences)))}}

    print(f"  Found {len(clusters)} topic clusters")

    # ── Step 5: Build topic objects ───────────────────────────────
    total_sents = len(sentences)
    raw_topics  = []

    for label, data in clusters.items():
        cluster_sents = data['sentences']
        cluster_idxs  = data['indices']

        # Get embeddings for this cluster
        cluster_embs = all_embeddings[cluster_idxs]

        # Position = average position of sentences in document
        position = np.mean(cluster_idxs) / total_sents

        # Extract title
        title = extract_topic_title(cluster_sents)

        # Extract full content
        content = ' '.join(cluster_sents)

        # Extract keywords
        keywords = extract_keywords(content, top_n=8)

        # Detect subtopics
        subtopics_raw = detect_subtopics(cluster_sents, cluster_embs)

        # Score
        score = score_topic_semantic(
            cluster_sents, cluster_embs,
            all_embeddings, position, total_sents
        )

        raw_topics.append({
            'title'    : title,
            'content'  : content,
            'keywords' : keywords,
            'subtopics': subtopics_raw,
            'score'    : score,
            'position' : position,
        })

    # ── Step 6: Assign priorities ─────────────────────────────────
    all_scores = [t['score'] for t in raw_topics]

    result = []
    for t in raw_topics:
        priority = assign_priority(t['score'], all_scores)

        subtopic_objects = [
            Subtopic(
                title      = sub['title'],
                content    = sub['content'],
                word_count = sub['word_count'],
                keywords   = sub['keywords'],
            )
            for sub in t['subtopics']
        ]

        result.append(Topic(
            title      = t['title'],
            priority   = priority,
            score      = t['score'],
            word_count = len(t['content'].split()),
            keywords   = t['keywords'],
            subtopics  = subtopic_objects,
        ))

    # ── Step 7: Sort — High first, then by score ──────────────────
    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    result.sort(key=lambda t: (order[t.priority], -t.score))

    return result