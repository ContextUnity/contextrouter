"""RAG-specific types and schemas (citations, retrieval metadata).

These types are specific to the RAG implementation and should not be in the core.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

SourceType: TypeAlias = Literal["book", "video", "qa", "web", "knowledge", "unknown"]


# ---- UI Citation Schemas (camelCase) ----------------------------------------
# These mirror what the frontend expects after formatting.


class UICitationBase(TypedDict, total=False):
    type: NotRequired[str]
    title: NotRequired[str]
    relevance: NotRequired[float]


class UICitationVideo(UICitationBase, total=False):
    type: NotRequired[Literal["video"]]
    videoId: NotRequired[str | None]
    videoUrl: NotRequired[str | None]
    timestamp: NotRequired[str | None]
    timestampSeconds: NotRequired[int | None]
    keywords: NotRequired[list[str]]
    summary: NotRequired[str]
    quote: NotRequired[str]


class UICitationBook(UICitationBase, total=False):
    type: NotRequired[Literal["book"]]
    chapter: NotRequired[str]
    chapterNumber: NotRequired[int | None]
    pageStart: NotRequired[int | None]
    pageEnd: NotRequired[int | None]
    keywords: NotRequired[list[str]]
    quote: NotRequired[str]


class UICitationQA(UICitationBase, total=False):
    type: NotRequired[Literal["qa"]]
    question: NotRequired[str]
    answer: NotRequired[str]
    keywords: NotRequired[list[str]]


class UICitationWeb(UICitationBase, total=False):
    type: NotRequired[Literal["web"]]
    summary: NotRequired[str]
    url: NotRequired[str | None]


class UICitationUnknown(UICitationBase, total=False):
    type: NotRequired[Literal["unknown"]]


UICitation: TypeAlias = (
    UICitationVideo | UICitationBook | UICitationQA | UICitationWeb | UICitationUnknown
)


# ---- Retrieval & Brain Types (snake_case) -----------------------------------


class RetrievedDoc(TypedDict, total=False):
    """Normalized document dict emitted by retrievers (snake_case keys)."""

    # Common
    source_type: SourceType
    content: str
    text: str
    title: str
    snippet: str
    url: str
    keywords: list[str]
    relevance: float
    relevance_score: float

    # Book
    book_title: str
    chapter: str
    chapter_number: int
    page_start: int
    page_end: int
    quote: str

    # Video
    video_name: str
    video_id: str
    video_url: str
    timestamp: str
    timestamp_seconds: int
    summary: str

    # Q&A
    session_title: str
    question: str
    answer: str

    # Knowledge
    filename: str
    description: str


class RawCitation(TypedDict, total=False):
    """Raw citation dict produced by the brain (snake_case keys)."""

    source_type: SourceType
    title: str
    keywords: list[str]
    relevance: float

    # Book
    book_title: str
    chapter: str
    chapter_number: int
    page_start: int
    page_end: int
    quote: str

    # Video
    video_name: str
    video_id: str
    video_url: str
    timestamp: str
    timestamp_seconds: int
    summary: str

    # Q&A
    session_title: str
    question: str
    answer: str

    # Web
    url: str
    summary: str


class RuntimeRagSettings(TypedDict, total=False):
    """Per-user persisted runtime settings for RAG (host-owned)."""

    rag_dataset: Literal["blue", "green"]
    model_name: str
    max_books: int
    max_videos: int
    max_qa: int
    citations_books: int
    citations_videos: int
    citations_qa: int
    max_output_tokens: int
    temperature: float
    general_retrieval_enabled: bool
    general_retrieval_initial_count: int
    general_retrieval_final_count: int
    reranking_enabled: bool
    ranker_model: str
    provider: str
    dual_read_enabled: bool
    dual_read_shadow_backend: str
    dual_read_sample_rate: float
    dual_read_timeout_ms: int
    dual_read_log_payloads: bool
