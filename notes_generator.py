import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def generate_notes(summary: str, min_length: int = 20) -> list[str]:
    """Split summary into sentences and filter to meaningful ones."""
    sentences = nltk.tokenize.sent_tokenize(summary)
    return [s for s in sentences if len(s) >= min_length]
