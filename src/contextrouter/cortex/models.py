"""Pydantic models for cortex-side data entities.

Note: RAG document/citation entities live under `modules/retrieval/rag/models.py`.
This module re-exports them for backward compatibility.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class BaseEntity(BaseModel):
    """Base Pydantic model with common configuration."""

    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",  # Allow extra fields for flexibility during retrieval
    )


class IntentResult(BaseEntity):
    """Structured output from intent detection."""

    intent: str
    intent_text: str
    cleaned_query: str
    retrieval_queries: List[str] = Field(default_factory=list)
    user_language: str = ""
    ignore_history: bool = False
    taxonomy_concepts: List[str] = Field(default_factory=list)
    taxonomy_categories: List[str] = Field(default_factory=list)


# Backward-compatible re-exports for RAG models (used by callers and tests).
from contextrouter.modules.retrieval.rag.models import Citation, RetrievedDoc  # noqa: E402

__all__ = ["BaseEntity", "IntentResult", "Citation", "RetrievedDoc"]
