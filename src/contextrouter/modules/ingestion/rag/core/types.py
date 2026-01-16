"""Core data types for the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NotRequired, TypedDict

from contextrouter.core.types import StructDataValue

# ---- Ingestion Struct Data (snake_case) -------------------------------------
# These are the "struct_data" payloads stored in ShadowRecord and exported.

SourceType = str


class BaseStructData(TypedDict, total=False):
    source_type: SourceType
    title: str
    citation_label: str
    keywords: list[str]


class BookStructData(BaseStructData, total=False):
    source_type: NotRequired[Literal["book"]]
    book_title: str
    chapter: str
    chapter_number: int | None
    page_start: int
    page_end: int | None
    quote: str


class VideoStructData(BaseStructData, total=False):
    source_type: NotRequired[Literal["video"]]
    video_id: str
    video_url: str
    video_name: str
    timestamp: str
    timestamp_seconds: int | None
    quote: str
    summary: str


class QAStructData(BaseStructData, total=False):
    source_type: NotRequired[Literal["qa"]]
    session_title: str
    source_title: str
    speaker: str
    question: str
    answer: str
    summary: str


class WebStructData(BaseStructData, total=False):
    source_type: NotRequired[Literal["web"]]
    title: str
    url: str
    summary: str
    quote: str


class KnowledgeStructData(BaseStructData, total=False):
    source_type: NotRequired[Literal["knowledge"]]
    filename: str
    description: str


class IngestionMetadata(TypedDict, total=False):
    """Flexible metadata container for ingestion data.

    Keys are optional and vary by source type:
    - Video: video_id, video_title, video_url, timestamp_seconds
    - Book: book_title, chapter, page_number
    - Web/QA: url, title, session_title
    - Generic: summary, keywords
    """

    # Video specific
    video_id: str
    video_title: str
    video_url: str
    timestamp_seconds: int

    # Book specific
    book_title: str
    chapter: str
    page_number: int

    # Web/QA specific
    url: str
    title: str
    session_title: str

    # Generic
    summary: str
    keywords: list[str]
    keyphrases: list[dict[str, object]]
    keyphrase_texts: list[str]
    ner_entities: list[dict[str, object]]
    ner_entities_by_type: dict[str, list[dict[str, object]]]
    ner_entity_count: int


class GraphEnrichmentResult(TypedDict, total=False):
    """Result of a graph enrichment call used during ingestion."""

    keywords: list[str]
    summary: str
    parent_categories: list[str]


@dataclass
class RawData:
    """Intermediate object returned by loaders.

    This represents raw content before transformation and chunking.
    """

    content: str
    source_type: str  # "video", "book", "qa", "web", "knowledge"
    metadata: IngestionMetadata = field(default_factory=dict)


@dataclass
class ShadowRecord:
    """Final output object that maps to Vertex AI schema.

    Shadow Context contract (strict separation):
    - input_text: Dirty/searchable payload. May include enrichment such as taxonomy tags,
      graph neighbors, summaries, etc. This is for retrieval only and must not be shown
      directly in UI.
    - struct_data: Clean UI/presentation fields + minimal typed metadata. Human-readable
      text fields like `quote`/`answer` MUST NOT contain enrichment tags or shadow headers.

    NOTE: Prefer snake_case keys in `struct_data` because runtime retriever parsing and
    citations are snake_case-first (API transport maps to frontend types).
    """

    id: str
    input_text: str  # The "Shadow" text (Content + AI Keywords + Graph Context)
    struct_data: dict[str, StructDataValue]  # Must match frontend citation schema
    citation_label: str | None = None
    title: str | None = None
    source_type: str | None = None
