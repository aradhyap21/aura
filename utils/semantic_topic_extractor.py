"""
AURA v4 — Semantic Topic Extractor
===================================

How real AI systems (NotebookLM, ChatGPT) handle documents:

STEP 1 — CLEAN
    Remove noise: slide numbers, professor names, dates, headers, footers.
    Only keep real educational sentences.

STEP 2 — EMBED
    Convert every sentence into a 384-dimensional meaning vector.
    Two sentences that mean the same thing → similar vectors.
    Sentence Transformers: all-MiniLM-L6-v2 (same model HuggingFace uses).

STEP 3 — CLUSTER
    Group sentences by meaning using HDBSCAN or KMeans.
    Each cluster = one topic. No headings needed.
    Works on raw lecture notes, slides, textbooks, anything.

STEP 4 — MERGE
    Similar clusters (cosine similarity > 0.82) → merged into one topic.
    Prevents duplicate topics like "Virtualization" and "Virtual Machines".

STEP 5 — FILTER NOISE
    Discard clusters that are:
    - Too short (< 40 real words)
    - Mostly metadata (dates, names, page numbers)
    - Low information density

STEP 6 — GENERATE TITLES
    Find the most central sentence in each cluster (closest to centroid).
    Clean it into a proper readable title.
    This is how NotebookLM names its topics.

STEP 7 — EXTRACT SUBTOPICS
    Sub-cluster each topic into 2-4 subtopics.
    Each subtopic gets its own clean title and keywords.

STEP 8 — SCORE & RANK
    4-signal scoring:
    - Information density (unique concepts per sentence)
    - Cluster coherence (how tightly grouped the sentences are)
    - Document coverage (proportion of document this topic covers)
    - Centrality (how central this topic is to the whole document)

STEP 9 — ASSIGN PRIORITY
    Top 20% by score → HIGH
    Next 35%         → MEDIUM
    Rest             → LOW
"""

import re
import numpy as np
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from models.schemas import Topic, Subtopic, Priority

# ── NLTK ──────────────────────────────────────────────────────────
for _res, _name in [
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_name, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# ── Stopwords ─────────────────────────────────────────────────────
STOPWORDS = set(stopwords.words("english"))
STOPWORDS.update({
    "also", "may", "use", "used", "using", "one", "two", "three",
    "first", "second", "third", "example", "figure", "table",
    "chapter", "section", "page", "note", "following", "shown",
    "refer", "thus", "hence", "however", "therefore", "whereas",
    "slide", "introduction", "conclusion", "summary", "overview",
    "definition", "versus", "within", "between", "through",
    "said", "say", "well", "just", "can", "get", "make", "let",
    "will", "shall", "would", "could", "should", "must", "need",
    "given", "like", "many", "much", "every", "each", "such",
    "upon", "still", "even", "back", "next", "last", "new",
    "good", "great", "important", "key", "main", "major",
    "provide", "provides", "provided", "including", "includes",
    "ensure", "ensures", "enables", "allows", "allows",
    "different", "various", "several", "specific", "particular",
    "current", "modern", "simple", "complex", "large", "small",
})

# ── Noise patterns (metadata, not educational content) ────────────
NOISE_PATTERNS = [
    r'^\s*\d{1,3}\s*$',                          # page numbers
    r'prepared\s+by',                             # "prepared by Dr..."
    r'\d{1,2}/\d{1,2}/\d{2,4}',                  # dates
    r'^(unit|chapter|module|lecture)\s*\d',       # unit headers
    r'^slide\s*\d',                               # slide numbers
    r'^\s*---\s*slide\s*\d',                      # slide separators
    r'^\s*\[notes?:',                             # notes labels
    r'copyright|all rights reserved',             # copyright
    r'^\s*[a-z]\s*$',                             # single letters
    r'www\.|http',                                # URLs
    r'@[a-zA-Z]',                                 # emails
]

_NOISE_RE = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — CLEAN & SPLIT
# ═══════════════════════════════════════════════════════════════════

def is_noise_sentence(s: str) -> bool:
    """Return True if sentence is metadata/noise, not content."""
    s = s.strip()
    # Too short
    if len(s.split()) < 5:
        return True
    # Mostly numbers/punctuation
    alpha = sum(c.isalpha() for c in s)
    if alpha / max(len(s), 1) < 0.5:
        return True
    # Matches noise patterns
    for pattern in _NOISE_RE:
        if pattern.search(s):
            return True
    return False


def split_and_clean(text: str) -> list[str]:
    """
    Split document into clean sentences.
    Remove all noise: metadata, headers, footers, slide labels.
    """
    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove slide separators
    text = re.sub(r'---\s*Slide\s*\d+\s*---', '\n', text, flags=re.IGNORECASE)

    sentences = []
    for para in text.split('\n'):
        para = para.strip()
        if not para or len(para) < 20:
            continue
        try:
            sents = sent_tokenize(para)
        except Exception:
            sents = [para]
        for s in sents:
            s = s.strip()
            if not is_noise_sentence(s):
                sentences.append(s)

    return sentences


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — EMBED
# ═══════════════════════════════════════════════════════════════════

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("  Loading embedding model...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Embedding model ready.")
        except ImportError:
            print("  ⚠ sentence-transformers not found. Using TF-IDF fallback.")
            _embedder = "tfidf"
    return _embedder


def embed(sentences: list[str]) -> np.ndarray:
    """Embed sentences into meaning vectors."""
    model = get_embedder()

    if model == "tfidf":
        vec = TfidfVectorizer(max_features=300, stop_words='english')
        try:
            return vec.fit_transform(sentences).toarray().astype(np.float32)
        except Exception:
            return np.random.rand(len(sentences), 50).astype(np.float32)

    return model.encode(
        sentences,
        batch_size        = 64,
        show_progress_bar = False,
        convert_to_numpy  = True,
        normalize_embeddings = True,  # L2 normalize → cosine similarity = dot product
    )


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — CLUSTER
# ═══════════════════════════════════════════════════════════════════

def cluster(embeddings: np.ndarray, n_sentences: int) -> np.ndarray:
    """
    Cluster sentence embeddings into topic groups.
    Tries HDBSCAN → UMAP+KMeans → KMeans fallback.
    """

    # ── Try HDBSCAN (best quality, handles noise natively) ─────────
    try:
        import hdbscan
        min_size = max(3, n_sentences // 25)
        clf = hdbscan.HDBSCAN(
            min_cluster_size         = min_size,
            min_samples              = 2,
            metric                   = 'euclidean',
            cluster_selection_method = 'eom',
        )
        labels = clf.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 3:
            return labels
    except ImportError:
        pass

    # ── Try UMAP + KMeans ──────────────────────────────────────────
    try:
        import umap
        from sklearn.cluster import KMeans

        n_comp = min(15, embeddings.shape[1], n_sentences - 2)
        reducer = umap.UMAP(
            n_components = n_comp,
            n_neighbors  = max(5, min(15, n_sentences // 8)),
            min_dist     = 0.05,
            metric       = 'cosine',
            random_state = 42,
        )
        reduced    = reducer.fit_transform(embeddings)
        n_clusters = max(4, min(18, n_sentences // 12))
        km         = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        return km.fit_predict(reduced)
    except ImportError:
        pass

    # ── KMeans fallback (always available) ─────────────────────────
    from sklearn.cluster import KMeans
    n_clusters = max(4, min(18, n_sentences // 12))
    km         = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    return km.fit_predict(embeddings)


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — MERGE SIMILAR CLUSTERS
# ═══════════════════════════════════════════════════════════════════

def merge_similar_clusters(
    clusters: dict,
    embeddings: np.ndarray,
    threshold: float = 0.82,
) -> dict:
    """
    Merge clusters whose centroids are very similar.
    This prevents duplicate topics like:
    'Virtualization Virtual Resources' and 'Virtual Operating Hypervisor'
    """
    ids = list(clusters.keys())
    if len(ids) <= 2:
        return clusters

    # Compute centroid for each cluster
    centroids = {}
    for cid, data in clusters.items():
        idxs          = data['indices']
        centroids[cid] = embeddings[idxs].mean(axis=0)

    centroid_matrix = np.array([centroids[i] for i in ids])
    centroid_matrix = normalize(centroid_matrix)

    # Compute pairwise similarity
    sim_matrix = cosine_similarity(centroid_matrix)

    merged_into = {i: i for i in ids}  # maps cluster → its merged parent

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            # Resolve current parents
            pa = merged_into[a]
            pb = merged_into[b]
            if pa == pb:
                continue
            if sim_matrix[i][j] >= threshold:
                # Merge smaller into larger
                if len(clusters[pa]['sentences']) >= len(clusters[pb]['sentences']):
                    merged_into[b] = pa
                    merged_into[pb] = pa
                else:
                    merged_into[a] = pb
                    merged_into[pa] = pb

    # Build merged clusters
    merged = {}
    for cid, data in clusters.items():
        parent = merged_into[cid]
        if parent not in merged:
            merged[parent] = {'sentences': [], 'indices': []}
        merged[parent]['sentences'].extend(data['sentences'])
        merged[parent]['indices'].extend(data['indices'])

    return merged


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — FILTER NOISE CLUSTERS
# ═══════════════════════════════════════════════════════════════════

def is_noise_cluster(sentences: list[str]) -> bool:
    """
    Return True if a cluster is mostly noise/metadata.
    We discard it.
    """
    # Too few sentences
    if len(sentences) < 2:
        return True

    full_text = ' '.join(sentences)
    words     = re.findall(r'\b[a-zA-Z]{3,}\b', full_text.lower())
    real_words = [w for w in words if w not in STOPWORDS]

    # Too few real content words
    if len(real_words) < 30:
        return True

    # Mostly proper nouns / names (metadata)
    # Count words that are capitalised mid-sentence
    caps_words = re.findall(r'(?<!\. )\b[A-Z][a-z]+\b', full_text)
    if len(caps_words) > 0 and len(caps_words) / max(len(words), 1) > 0.6:
        return True

    return False


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — GENERATE TITLES
# ═══════════════════════════════════════════════════════════════════


# Known acronyms that should be uppercased in titles
ACRONYMS = {
    'ngc', 'nvidia', 'api', 'ai', 'ml', 'os', 'vms', 'vm', 'vdi',
    'sdlc', 'cli', 'ci', 'cd', 'hpc', 'aws', 'gcp', 'sdk', 'nfv',
    'sdn', 'qa', 'gpu', 'cpu', 'ram', 'iot', 'devops', 'docker',
    'kubernetes', 'linux', 'windows', 'vmware', 'hyper-v',
}


def fix_acronyms(text: str) -> str:
    """Uppercase known technical acronyms in a title."""
    words = text.split()
    result = []
    for w in words:
        lower = w.lower().rstrip('s')  # handle plurals like "vms"
        if w.lower() in ACRONYMS or lower in ACRONYMS:
            result.append(w.upper())
        else:
            result.append(w)
    return ' '.join(result)


def generate_title(sentences: list[str], embeddings: np.ndarray) -> str:
    """
    Generate a clean 2-4 word topic title directly from TF-IDF keywords.

    The keywords are the ground truth — they are already the most
    distinctive terms in the cluster. Building the title from them
    directly gives clean results like:
      "Docker Hub Container Registry"
      "Virtualization Technology"
      "NVIDIA NGC Cloud Platform"
      "System Software Engineering"
    """
    if not sentences:
        return "Unknown Topic"

    # Get top TF-IDF terms — boost longer ngrams (more specific)
    try:
        vec = TfidfVectorizer(
            max_features  = 60,
            ngram_range   = (1, 3),
            stop_words    = 'english',
            token_pattern = r'\b[a-zA-Z][a-zA-Z\-]{1,}\b',
        )
        matrix = vec.fit_transform(sentences)
        names  = vec.get_feature_names_out()
        scores = np.asarray(matrix.mean(axis=0)).flatten()

        boosted = []
        for name, score in zip(names, scores):
            n_words = len(name.split())
            boost   = 1.0 + (n_words - 1) * 0.6
            if not all(w in STOPWORDS for w in name.split()):
                boosted.append((name, score * boost))

        top_terms = [n for n, _ in sorted(boosted, key=lambda x: x[1], reverse=True)][:8]

    except Exception:
        words     = re.findall(r'\b[a-zA-Z]{4,}\b', ' '.join(sentences).lower())
        filtered  = [w for w in words if w not in STOPWORDS]
        top_terms = [w for w, _ in Counter(filtered).most_common(8)]

    title = smart_keyword_title(top_terms)
    title = fix_acronyms(title)
    return title


def smart_keyword_title(top_terms: list[str]) -> str:
    """
    Build a clean title from top TF-IDF keywords.
    Prefer bigrams (more specific) over single words.
    """
    bigrams  = [t for t in top_terms if ' ' in t]
    unigrams = [t for t in top_terms if ' ' not in t]

    if bigrams:
        # Use best bigram as base
        base  = bigrams[0]
        # Add one more unigram if available and not already in base
        extra = next((u for u in unigrams if u.lower() not in base.lower()), None)
        title = f"{base} {extra}" if extra else base
    elif unigrams:
        title = ' '.join(unigrams[:3])
    else:
        return "General Topic"

    return title_case(title)


def title_case(text: str) -> str:
    """
    Proper title case — capitalize meaningful words,
    keep articles/prepositions lowercase unless first word.
    """
    LOWER_WORDS = {'a', 'an', 'the', 'and', 'but', 'or', 'for',
                   'nor', 'on', 'at', 'to', 'by', 'in', 'of', 'with'}
    words  = text.strip().split()
    result = []
    for i, w in enumerate(words):
        if i == 0 or w.lower() not in LOWER_WORDS:
            result.append(w.capitalize())
        else:
            result.append(w.lower())
    return ' '.join(result)


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — EXTRACT KEYWORDS
# ═══════════════════════════════════════════════════════════════════

def extract_keywords(sentences: list[str], top_n: int = 8) -> list[str]:
    """
    Extract meaningful keywords using TF-IDF across sentences.
    Prefers multi-word technical terms (bigrams).
    """
    if len(sentences) < 2:
        words    = re.findall(r'\b[a-zA-Z]{4,}\b', ' '.join(sentences).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]

    try:
        vec = TfidfVectorizer(
            max_features  = 300,
            ngram_range   = (1, 2),
            stop_words    = 'english',
            token_pattern = r'\b[a-zA-Z][a-zA-Z\-]{2,}\b',
            min_df        = 1,
        )
        matrix = vec.fit_transform(sentences)
        names  = vec.get_feature_names_out()
        scores = np.asarray(matrix.mean(axis=0)).flatten()

        # Boost bigrams (more informative than single words)
        boosted = []
        for name, score in zip(names, scores):
            if ' ' in name:
                boosted.append((name, score * 1.4))
            else:
                boosted.append((name, score))

        ranked = sorted(boosted, key=lambda x: x[1], reverse=True)

        # Filter out stopword-only terms
        clean = []
        for term, _ in ranked:
            words = term.split()
            if all(w in STOPWORDS for w in words):
                continue
            clean.append(term)
            if len(clean) == top_n:
                break

        return clean

    except Exception:
        words    = re.findall(r'\b[a-zA-Z]{4,}\b', ' '.join(sentences).lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return [w for w, _ in Counter(filtered).most_common(top_n)]


# ═══════════════════════════════════════════════════════════════════
# STEP 8 — DETECT SUBTOPICS
# ═══════════════════════════════════════════════════════════════════

def detect_subtopics(
    sentences  : list[str],
    embeddings : np.ndarray,
) -> list[dict]:
    """
    Within a topic, find subtopics by sub-clustering.
    Only for topics with 8+ sentences.
    Each subtopic gets a proper title from its most central sentence.
    """
    if len(sentences) < 8:
        return []

    try:
        from sklearn.cluster import AgglomerativeClustering

        n_sub     = max(2, min(4, len(sentences) // 6))
        normed    = normalize(embeddings)
        sub_model = AgglomerativeClustering(
            n_clusters = n_sub,
            metric     = 'cosine',
            linkage    = 'average',
        )
        sub_labels = sub_model.fit_predict(normed)

        subtopics = []
        for sub_id in sorted(set(sub_labels)):
            idxs      = [i for i, l in enumerate(sub_labels) if l == sub_id]
            sub_sents = [sentences[i] for i in idxs]
            sub_embs  = embeddings[idxs]

            if len(sub_sents) < 2:
                continue

            sub_content = ' '.join(sub_sents)
            real_words  = re.findall(r'\b[a-zA-Z]{3,}\b', sub_content.lower())
            real_words  = [w for w in real_words if w not in STOPWORDS]

            if len(real_words) < 15:
                continue

            sub_title = generate_title(sub_sents, sub_embs)
            sub_kws   = extract_keywords(sub_sents, top_n=5)

            subtopics.append({
                'title'     : sub_title,
                'content'   : sub_content,
                'word_count': len(sub_content.split()),
                'keywords'  : sub_kws,
            })

        return subtopics

    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# STEP 9 — SCORE
# ═══════════════════════════════════════════════════════════════════

def score_topic(
    sentences      : list[str],
    embeddings     : np.ndarray,
    all_embeddings : np.ndarray,
    position       : float,
) -> float:
    """
    4-signal scoring:

    1. Information Density  — unique concepts per sentence
    2. Cluster Coherence    — how tightly clustered (high = focused topic)
    3. Document Coverage    — proportion of document this topic covers
    4. Centrality           — how central to whole document meaning
    """
    n = len(sentences)

    # Signal 1: Information Density
    text       = ' '.join(sentences)
    all_words  = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    real_words = [w for w in all_words if w not in STOPWORDS]
    unique     = len(set(real_words))
    density    = min(unique / max(n * 3, 1), 1.0)

    # Signal 2: Coherence — avg similarity to centroid
    centroid  = embeddings.mean(axis=0, keepdims=True)
    sims      = cosine_similarity(embeddings, centroid).flatten()
    coherence = float(sims.mean())

    # Signal 3: Coverage
    coverage = min(n / max(len(all_embeddings) * 0.04, 1), 1.0)

    # Signal 4: Centrality — how similar this cluster is to whole doc mean
    doc_centroid  = all_embeddings.mean(axis=0, keepdims=True)
    cluster_cent  = normalize(centroid)
    doc_cent_norm = normalize(doc_centroid)
    centrality    = float(cosine_similarity(cluster_cent, doc_cent_norm)[0][0])

    score = (
        0.30 * density    +
        0.25 * coherence  +
        0.25 * coverage   +
        0.20 * centrality
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


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def parse_topics(raw_text: str) -> list:
    """
    Full 9-step semantic pipeline.
    Called by analyze_service.py.
    Returns clean Topic objects with subtopics, keywords, priority.
    """

    # ── Step 1: Clean & Split ─────────────────────────────────────
    sentences = split_and_clean(raw_text)
    print(f"  {len(sentences)} clean sentences after noise removal")

    if len(sentences) < 5:
        return []

    # ── Step 2: Embed ─────────────────────────────────────────────
    print(f"  Embedding {len(sentences)} sentences...")
    all_embeddings = embed(sentences)

    # ── Step 3: Cluster ───────────────────────────────────────────
    print(f"  Clustering...")
    labels = cluster(all_embeddings, len(sentences))

    # ── Group sentences by cluster ────────────────────────────────
    raw_clusters = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        if label not in raw_clusters:
            raw_clusters[label] = {'sentences': [], 'indices': []}
        raw_clusters[label]['sentences'].append(sentences[i])
        raw_clusters[label]['indices'].append(i)

    if not raw_clusters:
        raw_clusters = {0: {'sentences': sentences, 'indices': list(range(len(sentences)))}}

    print(f"  {len(raw_clusters)} raw clusters found")

    # ── Step 4: Merge similar clusters ───────────────────────────
    merged_clusters = merge_similar_clusters(raw_clusters, all_embeddings, threshold=0.82)
    print(f"  {len(merged_clusters)} clusters after merging")

    # ── Step 5: Filter noise clusters ────────────────────────────
    clean_clusters = {
        cid: data for cid, data in merged_clusters.items()
        if not is_noise_cluster(data['sentences'])
    }
    print(f"  {len(clean_clusters)} clusters after noise filtering")

    if not clean_clusters:
        clean_clusters = merged_clusters

    # ── Steps 6-9: Build Topic objects ────────────────────────────
    total_sents = len(sentences)
    raw_topics  = []

    for cid, data in clean_clusters.items():
        sents  = data['sentences']
        idxs   = data['indices']
        embs   = all_embeddings[idxs]

        # Position = where in document (0=start, 1=end)
        position = np.mean(idxs) / total_sents

        # Step 6: Title from most central sentence
        title = generate_title(sents, embs)

        # Keywords
        keywords = extract_keywords(sents, top_n=8)

        # Step 7: Subtopics
        subtopics_raw = detect_subtopics(sents, embs)

        # Content = full clean sentences joined
        content = ' '.join(sents)

        # Step 8: Score
        score = score_topic(sents, embs, all_embeddings, position)

        raw_topics.append({
            'title'    : title,
            'content'  : content,
            'keywords' : keywords,
            'subtopics': subtopics_raw,
            'score'    : score,
        })

    # ── Step 9: Assign priorities ─────────────────────────────────
    all_scores = [t['score'] for t in raw_topics]

    result = []
    for t in raw_topics:
        priority = assign_priority(t['score'], all_scores)

        subtopic_objects = [
            Subtopic(
                title      = sub['title'],
                content    = sub['content'][:800],
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

    # Sort: High first, then by score descending
    order = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
    result.sort(key=lambda t: (order[t.priority], -t.score))

    print(f"  ✅ {len(result)} final topics extracted")
    return result