"""
AURA v2 — Schemas
Clean, focused data models for document analysis.
"""

from pydantic import BaseModel
from typing import List
from enum import Enum


class Priority(str, Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


# ── Subtopic under a main topic ───────────────────────────────────
class Subtopic(BaseModel):
    title       : str
    content     : str          # actual text under this subtopic
    word_count  : int
    keywords    : List[str]    # top keywords extracted from this section


# ── Main topic (chapter/section heading) ─────────────────────────
class Topic(BaseModel):
    title       : str
    priority    : Priority
    score       : float
    word_count  : int
    keywords    : List[str]    # top keywords from entire topic section
    subtopics   : List[Subtopic]


# ── Full extraction result ────────────────────────────────────────
class ExtractionResult(BaseModel):
    file_name        : str
    file_type        : str
    total_words      : int
    total_topics     : int
    total_subtopics  : int
    high_count       : int
    medium_count     : int
    low_count        : int
    warnings         : List[str]
    topics           : List[Topic]   # sorted: High first, then Medium, then Low"""
"""
AURA v2 — Schemas
Clean, focused data models for document analysis.
"""

from pydantic import BaseModel
from typing import List
from enum import Enum


class Priority(str, Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


# ── Subtopic under a main topic ───────────────────────────────────
class Subtopic(BaseModel):
    title       : str
    content     : str          # actual text under this subtopic
    word_count  : int
    keywords    : List[str]    # top keywords extracted from this section


# ── Main topic (chapter/section heading) ─────────────────────────
class Topic(BaseModel):
    title       : str
    priority    : Priority
    score       : float
    word_count  : int
    keywords    : List[str]    # top keywords from entire topic section
    subtopics   : List[Subtopic]


# ── Full extraction result ────────────────────────────────────────
class ExtractionResult(BaseModel):
    file_name        : str
    file_type        : str
    total_words      : int
    total_topics     : int
    total_subtopics  : int
    high_count       : int
    medium_count     : int
    low_count        : int
    warnings         : List[str]
    topics           : List[Topic]   # sorted: High first, then Medium, then Low"""
"""
AURA v2 — Schemas
Clean, focused data models for document analysis.
"""

from pydantic import BaseModel
from typing import List
from enum import Enum


class Priority(str, Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


# ── Subtopic under a main topic ───────────────────────────────────
class Subtopic(BaseModel):
    title       : str
    content     : str          # actual text under this subtopic
    word_count  : int
    keywords    : List[str]    # top keywords extracted from this section


# ── Main topic (chapter/section heading) ─────────────────────────
class Topic(BaseModel):
    title       : str
    priority    : Priority
    score       : float
    word_count  : int
    keywords    : List[str]    # top keywords from entire topic section
    subtopics   : List[Subtopic]


# ── Full extraction result ────────────────────────────────────────
class ExtractionResult(BaseModel):
    file_name        : str
    file_type        : str
    total_words      : int
    total_topics     : int
    total_subtopics  : int
    high_count       : int
    medium_count     : int
    low_count        : int
    warnings         : List[str]
    topics           : List[Topic]   # sorted: High first, then Medium, then Low"""
"""
AURA v2 — Schemas
Clean, focused data models for document analysis.
"""

from pydantic import BaseModel
from typing import List
from enum import Enum


class Priority(str, Enum):
    HIGH   = "High"
    MEDIUM = "Medium"
    LOW    = "Low"


# ── Subtopic under a main topic ───────────────────────────────────
class Subtopic(BaseModel):
    title       : str
    content     : str          # actual text under this subtopic
    word_count  : int
    keywords    : List[str]    # top keywords extracted from this section


# ── Main topic (chapter/section heading) ─────────────────────────
class Topic(BaseModel):
    title       : str
    priority    : Priority
    score       : float
    word_count  : int
    keywords    : List[str]    # top keywords from entire topic section
    subtopics   : List[Subtopic]


# ── Full extraction result ────────────────────────────────────────
class ExtractionResult(BaseModel):
    file_name        : str
    file_type        : str
    total_words      : int
    total_topics     : int
    total_subtopics  : int
    high_count       : int
    medium_count     : int
    low_count        : int
    warnings         : List[str]
    topics           : List[Topic]   # sorted: High first, then Medium, then Low