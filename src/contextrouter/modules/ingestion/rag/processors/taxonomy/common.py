from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ...core.utils import (
    normalize_ambiguous_unicode,
    strip_markdown_from_text,
)

DEDUP_PREFIX_CHARS = 120

_PAGE_MARKER_PATTERNS: tuple[tuple[str, int], ...] = (
    (r"<page:\s*\d+>", re.IGNORECASE),
    (r"<page:\d+>", re.IGNORECASE),
    (r"---\s*end\s+of\s+page[^\n]*---", re.IGNORECASE),
)


@dataclass(frozen=True)
class TaxonomySample:
    text: str
    source_type: str
    doc_key: str


def stable_hash_u64(s: str) -> int:
    """Deterministic hash used for sampling (order-independent)."""
    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def clean_for_taxonomy_sample(text: str) -> str:
    """Deterministic cleaning for taxonomy extraction."""
    s = normalize_ambiguous_unicode(text or "")
    for pat, flags in _PAGE_MARKER_PATTERNS:
        s = re.sub(pat, " ", s, flags=flags)
    s = strip_markdown_from_text(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def windowed_snippets(text: str, *, window_chars: int, max_windows: int) -> list[str]:
    """Deterministically sample multiple windows from long content."""
    s = (text or "").strip()
    if not s:
        return []
    if window_chars <= 0:
        return [s]
    if len(s) <= window_chars:
        return [s]

    k = max(1, int(max_windows))
    if k == 1:
        return [s[:window_chars]]

    # Evenly spaced windows from start -> end.
    max_start = max(0, len(s) - window_chars)
    if max_start == 0:
        return [s[:window_chars]]
    step = max(1, max_start // (k - 1))

    out: list[str] = []
    seen: set[str] = set()
    for i in range(k):
        start = min(max_start, i * step)
        chunk = s[start : start + window_chars].strip()
        if not chunk:
            continue
        key = chunk[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
        if len(out) >= k:
            break
    return out
