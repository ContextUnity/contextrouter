"""Lightweight MMR selection to reduce near-duplicates."""

from __future__ import annotations

import re

from .models import RetrievedDoc

_token_re = re.compile(r"[a-zA-Z0-9_]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _token_re.findall(text or "") if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def mmr_select(
    *,
    query: str,
    candidates: list[RetrievedDoc],
    k: int,
    lambda_mult: float,
) -> list[RetrievedDoc]:
    if k <= 0 or not candidates:
        return []
    if k >= len(candidates):
        return candidates

    query_tokens = _tokens(query)
    doc_tokens = [_tokens((d.title or "") + " " + (d.content or "")) for d in candidates]

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < k:
        best_idx = None
        best_score = None
        for idx in list(remaining):
            relevance = _jaccard(query_tokens, doc_tokens[idx])
            diversity = 0.0
            if selected:
                diversity = max(_jaccard(doc_tokens[idx], doc_tokens[s]) for s in selected)
            score = (lambda_mult * relevance) - ((1 - lambda_mult) * diversity)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


__all__ = ["mmr_select"]
