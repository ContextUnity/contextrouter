"""Tests for RAG models, parity harness, and settings.

Zero-mock tests for:
- models.py: RetrievedDoc and Citation schema validation
- parity.py: _doc_key, _overlap, ParityConfig, should_run
- settings.py: RagRetrievalSettings defaults and validation
"""

from __future__ import annotations

from contextunity.router.modules.retrieval.rag.models import RetrievedDoc
from contextunity.router.modules.retrieval.rag.parity import (
    _doc_key,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _doc(
    source_type: str = "book",
    content: str = "test",
    **kwargs,
) -> RetrievedDoc:
    return RetrievedDoc(source_type=source_type, content=content, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# RetrievedDoc model
# ═══════════════════════════════════════════════════════════════════════


class TestRetrievedDoc:
    """Schema validation for RetrievedDoc Pydantic model."""

    def test_metadata_dict(self):
        doc = _doc(metadata={"source": "brain", "collection": "default"})
        assert doc.metadata["source"] == "brain"


# ═══════════════════════════════════════════════════════════════════════
# Citation model
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Parity harness — pure functions
# ═══════════════════════════════════════════════════════════════════════


class TestDocKey:
    """_doc_key creates stable dedup keys."""

    def test_url_preferred(self):
        doc = _doc(url="https://example.com/page")
        assert _doc_key(doc) == "https://example.com/page"

    def test_url_stripped(self):
        doc = _doc(url="  https://example.com  ")
        assert _doc_key(doc) == "https://example.com"

    def test_fallback_hash_without_url(self):
        doc = _doc(title="My Doc", content="Some content")
        key = _doc_key(doc)
        # Should be a sha256 hex digest (64 chars)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


# ═══════════════════════════════════════════════════════════════════════
# RagRetrievalSettings — defaults
# ═══════════════════════════════════════════════════════════════════════
