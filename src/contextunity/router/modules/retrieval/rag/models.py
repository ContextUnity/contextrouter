"""RAG-specific runtime models.
These models exist to support the "RAG with citations" capability.
They MUST NOT leak into `contextunity.router.core` (knowledge-agnostic kernel).
"""

from __future__ import annotations

from typing import ClassVar

from contextunity.core.types import JsonDict
from pydantic import BaseModel, ConfigDict, Field

from .types import SourceType


class BaseEntity(BaseModel):
    """Pydantic base for RAG data models: allows field aliases and ignores unknown keys."""

    model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True, extra="ignore")


class RetrievedDoc(BaseEntity):
    """Normalized document returned by RAG retrievers/providers/connectors."""

    # Minimal agnostic contract
    source_type: SourceType
    content: str
    relevance: float = 0.0
    # Keep metadata JSON-safe but avoid recursive type aliases in Pydantic schema generation.
    metadata: JsonDict = Field(default_factory=dict)

    # Optional presentation fields (commonly used by RAG UIs)
    title: str | None = None
    url: str | None = None
    snippet: str | None = None
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None
    quote: str | None = None

    # Back-compat: source-specific fields used by existing RAG citation builders
    book_title: str | None = None
    chapter: str | None = None
    chapter_number: int | None = None
    page_start: float | None = None
    page_end: float | None = None

    video_id: str | None = None
    video_url: str | None = None
    video_name: str | None = None
    timestamp: str | None = None
    timestamp_seconds: float | None = None

    session_title: str | None = None
    question: str | None = None
    answer: str | None = None


class Citation(BaseEntity):
    """UI-facing citation emitted by the RAG capability."""

    source_type: SourceType
    title: str
    content: str
    relevance: float = 0.0
    # Keep metadata JSON-safe but avoid recursive type aliases in Pydantic schema generation.
    metadata: JsonDict = Field(default_factory=dict)

    url: str | None = None
    keywords: list[str] = Field(default_factory=list)
    summary: str | None = None
    quote: str | None = None

    # Back-compat: source-specific fields
    video_id: str | None = None
    video_url: str | None = None
    timestamp: str | None = None
    timestamp_seconds: float | None = None
    page_start: float | None = None
    page_end: float | None = None

    book_title: str | None = None
    chapter: str | None = None
    chapter_number: int | None = None
    session_title: str | None = None
    question: str | None = None
    answer: str | None = None


__all__ = ["RetrievedDoc", "Citation"]
