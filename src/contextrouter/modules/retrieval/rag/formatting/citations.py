"""Citation formatter for UI schemas.

Converts Citation Pydantic models (snake_case) to canonical UI citation schema (camelCase).
Logic is intentionally preserved 1:1 from `contextrouter.integrations.citations.formatter`.
"""

from __future__ import annotations

import html
import re
from typing import Iterable

from ..models import Citation
from ..types import UICitation


def _clean_text(text: str | None) -> str:
    """Clean text for UI display: decode HTML entities, strip markdown artifacts."""
    if not text:
        return ""
    # Decode HTML entities (&#39; -> ', &amp; -> &, etc.)
    s = html.unescape(text)
    # Remove stray markdown header markers (## at random positions)
    s = re.sub(r"\s*#{2,}\s*", " ", s)
    # Normalize whitespace
    return " ".join(s.split()).strip()


def _coerce_timestamp_seconds(citation: Citation) -> int | None:
    """Coerce timestamp to seconds integer."""
    if citation.timestamp_seconds is not None:
        if citation.timestamp_seconds < 0:
            return None
        return int(citation.timestamp_seconds)

    # Fallback: parse string timestamp format (HH:MM:SS)
    ts = citation.timestamp
    if not isinstance(ts, str) or not ts.strip():
        return None

    try:
        parts = ts.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        if len(parts) == 1:
            return int(parts[0])
    except Exception:
        return None

    return None


def _coerce_page(value: float | None) -> int | None:
    """Coerce page number to integer."""
    if value is not None:
        return int(value)
    return None


def format_citations_to_ui(
    raw_citations: list[Citation],
    *,
    allowed_types: Iterable[str] | None = None,
) -> list[UICitation]:
    """Convert brain Citation models to canonical UI citation schema (camelCase).

    Args:
        raw_citations: List of Citation Pydantic models
        allowed_types: Optional iterable of source types to include

    Returns:
        List of citation dicts in camelCase UI schema format
    """
    allowed = None
    if allowed_types is not None:
        allowed = {x.strip() for x in allowed_types if isinstance(x, str) and x.strip()}
        if not allowed:
            allowed = None

    out: list[UICitation] = []

    for c in raw_citations:
        if c is None:
            continue

        source_type = c.source_type
        relevance = c.relevance

        if source_type == "video":
            if allowed is not None and "video" not in allowed:
                continue
            out.append(
                {
                    "type": "video",
                    "title": _clean_text(c.title) or "Video",
                    "videoId": c.video_id,
                    "videoUrl": c.video_url,
                    "timestamp": c.timestamp,
                    "timestampSeconds": _coerce_timestamp_seconds(c),
                    "keywords": c.keywords or [],
                    "summary": _clean_text(c.summary),
                    "quote": _clean_text(c.quote or c.content),
                    "relevance": relevance,
                }
            )
            continue

        if source_type == "book":
            if allowed is not None and "book" not in allowed:
                continue
            out.append(
                {
                    "type": "book",
                    "title": _clean_text(c.book_title or c.title) or "Unknown Book",
                    "chapter": _clean_text(c.chapter),
                    "chapterNumber": c.chapter_number,
                    "pageStart": _coerce_page(c.page_start),
                    "pageEnd": _coerce_page(c.page_end),
                    "keywords": c.keywords or [],
                    "quote": _clean_text(c.quote or c.content),
                }
            )
            continue

        if source_type == "qa":
            if allowed is not None and "qa" not in allowed:
                continue
            out.append(
                {
                    "type": "qa",
                    "title": _clean_text(c.session_title or c.title) or "Q&A Session",
                    "question": _clean_text(c.question),
                    "answer": _clean_text(c.answer or c.content),
                    "keywords": c.keywords or [],
                    "relevance": relevance,
                }
            )
            continue

        if source_type == "web":
            if allowed is not None and "web" not in allowed:
                continue
            out.append(
                {
                    "type": "web",
                    "title": _clean_text(c.title) or "Web",
                    "summary": _clean_text(c.summary or c.content),
                    "url": c.url,
                }
            )
            continue

        # Unknown source type
        out.append(
            {
                "type": "unknown",
                "title": _clean_text(c.title) or "Source",
                "relevance": relevance,
            }
        )

    return out


__all__ = ["format_citations_to_ui"]
