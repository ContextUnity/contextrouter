"""Lightweight MMR selection to reduce near-duplicates."""

from __future__ import annotations

from .models import RetrievedDoc


def _tokens(text: str) -> set[str]:
    """Tokenise *text* into a lowercase set of alphanumeric words (len > 2)."""
    tokens: set[str] = set()
    current: list[str] = []
    for ch in (text or "").lower():
        if ch.isalnum() or ch == "_":
            current.append(ch)
            continue
        if current:
            token = "".join(current)
            if len(token) > 2:
                tokens.add(token)
            current = []
    if current:
        token = "".join(current)
        if len(token) > 2:
            tokens.add(token)
    return tokens


def _jaccard(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity coefficient between token sets *a* and *b*."""
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
    """Greedily select up to *k* documents by Maximal Marginal Relevance using Jaccard token overlap."""
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
