"""RAG-specific runtime models.

These models exist to support the "RAG with citations" capability.
They MUST NOT leak into `contextrouter.core` (knowledge-agnostic kernel).
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .types import SourceType


class BaseEntity(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class RetrievedDoc(BaseEntity):
    """Normalized document returned by RAG retrievers/providers/connectors."""

    # Minimal agnostic contract
    source_type: SourceType
    content: str
    relevance: float = 0.0
    # Keep metadata JSON-safe but avoid recursive type aliases in Pydantic schema generation.
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Optional presentation fields (commonly used by RAG UIs)
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    quote: Optional[str] = None

    # Back-compat: source-specific fields used by existing RAG citation builders
    book_title: Optional[str] = None
    chapter: Optional[str] = None
    chapter_number: Optional[int] = None
    page_start: Optional[float] = None
    page_end: Optional[float] = None

    video_id: Optional[str] = None
    video_url: Optional[str] = None
    video_name: Optional[str] = None
    timestamp: Optional[str] = None
    timestamp_seconds: Optional[float] = None

    session_title: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


class Citation(BaseEntity):
    """UI-facing citation emitted by the RAG capability."""

    source_type: SourceType
    title: str
    content: str
    relevance: float = 0.0
    # Keep metadata JSON-safe but avoid recursive type aliases in Pydantic schema generation.
    metadata: dict[str, Any] = Field(default_factory=dict)

    url: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    quote: Optional[str] = None

    # Back-compat: source-specific fields
    video_id: Optional[str] = None
    video_url: Optional[str] = None
    timestamp: Optional[str] = None
    timestamp_seconds: Optional[float] = None
    page_start: Optional[float] = None
    page_end: Optional[float] = None

    book_title: Optional[str] = None
    chapter: Optional[str] = None
    chapter_number: Optional[int] = None
    session_title: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


__all__ = ["RetrievedDoc", "Citation"]
