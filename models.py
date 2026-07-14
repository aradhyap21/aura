from dataclasses import dataclass, field
from typing import List

VALID_BLOOM_LEVELS = ["Remember", "Understand", "Apply", "Analyze"]
VALID_PROMPT_TYPES = ["qa", "cloze"]


@dataclass
class ExtractedDocument:
    raw_text: str       # full extracted text before cleaning
    clean_text: str     # normalized text ready for ML
    page_count: int     # number of pages in source PDF

    def __post_init__(self):
        if not self.clean_text:
            raise ValueError("clean_text must be non-empty before passing to summarizer")


@dataclass
class QAPair:
    question: str
    answer: str
    bloom_level: str    # "Remember" | "Understand" | "Apply" | "Analyze"
    source_note: str    # the note sentence this was derived from

    def __post_init__(self):
        if self.bloom_level not in VALID_BLOOM_LEVELS:
            raise ValueError(
                f"bloom_level must be one of {VALID_BLOOM_LEVELS}, got '{self.bloom_level}'"
            )


@dataclass
class MCQItem:
    question: str
    options: List[str]  # exactly 4 options
    correct_index: int  # 0-3
    bloom_level: str

    def __post_init__(self):
        if len(self.options) != 4:
            raise ValueError(
                f"options must have exactly 4 elements, got {len(self.options)}"
            )
        if self.correct_index not in range(4):
            raise ValueError(
                f"correct_index must be in range [0, 3], got {self.correct_index}"
            )
        if self.bloom_level not in VALID_BLOOM_LEVELS:
            raise ValueError(
                f"bloom_level must be one of {VALID_BLOOM_LEVELS}, got '{self.bloom_level}'"
            )


@dataclass
class RecallPrompt:
    prompt: str
    expected: str
    prompt_type: str    # "qa" | "cloze"

    def __post_init__(self):
        if self.prompt_type not in VALID_PROMPT_TYPES:
            raise ValueError(
                f"prompt_type must be one of {VALID_PROMPT_TYPES}, got '{self.prompt_type}'"
            )


@dataclass
class StudyMaterial:
    summary: str
    notes: List[str]
    qa_pairs: List[QAPair]
    mcqs: List[MCQItem]
    diagram: str
    prompts: List[RecallPrompt]
