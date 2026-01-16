"""Dual-read parity harness for RAG retrieval providers."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from dataclasses import dataclass

from contextrouter.core import BiscuitToken
from contextrouter.modules.retrieval import BaseRetrievalPipeline

from .models import RetrievedDoc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParityConfig:
    enabled: bool
    shadow_backend: str | None
    sample_rate: float
    timeout_ms: int
    log_payloads: bool


@dataclass(frozen=True)
class ParityMetrics:
    overlap_at_k: float
    primary_count: int
    shadow_count: int
    primary_ms: float
    shadow_ms: float


def _doc_key(doc: RetrievedDoc) -> str:
    # Prefer stable URL, otherwise hash content+title to avoid logging raw text.
    raw = (doc.url or "").strip()
    if raw:
        return raw
    fallback = f"{doc.title or ''}::{doc.content or ''}"
    return hashlib.sha256(fallback.encode("utf-8")).hexdigest()


def _overlap(primary: list[RetrievedDoc], shadow: list[RetrievedDoc], k: int) -> float:
    if k <= 0:
        return 0.0
    p = {_doc_key(d) for d in primary[:k]}
    s = {_doc_key(d) for d in shadow[:k]}
    if not p:
        return 0.0
    return len(p & s) / max(1, min(len(p), len(s)))


class DualReadHarness:
    """Runs a shadow retrieval in parallel and logs parity metrics."""

    def __init__(self, cfg: ParityConfig) -> None:
        self._cfg = cfg
        self._base = BaseRetrievalPipeline()

    def should_run(self) -> bool:
        if not self._cfg.enabled:
            return False
        if not self._cfg.shadow_backend:
            return False
        return random.random() <= self._cfg.sample_rate

    async def _shadow_retrieve(
        self,
        *,
        query: str,
        token: BiscuitToken,
        limit: int,
    ) -> tuple[list[RetrievedDoc], float]:
        t0 = time.perf_counter()
        res = await self._base.execute(
            query,
            token=token,
            limit=limit,
            filters=None,
            providers=[self._cfg.shadow_backend] if self._cfg.shadow_backend else None,
        )
        docs: list[RetrievedDoc] = []
        for env in getattr(res, "envelopes", []) or []:
            content = getattr(env, "content", None)
            if isinstance(content, RetrievedDoc):
                docs.append(content)
            elif isinstance(content, dict):
                try:
                    docs.append(RetrievedDoc.model_validate(content))
                except Exception:
                    continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return docs, elapsed_ms

    async def compare(
        self,
        *,
        query: str,
        token: BiscuitToken,
        primary_docs: list[RetrievedDoc],
        primary_ms: float,
        limit: int,
    ) -> None:
        if not self.should_run():
            return

        timeout_s = max(0.05, self._cfg.timeout_ms / 1000.0)

        async def _run() -> tuple[list[RetrievedDoc], float]:
            return await asyncio.wait_for(
                self._shadow_retrieve(query=query, token=token, limit=limit),
                timeout=timeout_s,
            )

        try:
            shadow_docs, shadow_ms = await _run()
            overlap_at_k = _overlap(primary_docs, shadow_docs, k=min(limit, len(primary_docs)))
            metrics = ParityMetrics(
                overlap_at_k=overlap_at_k,
                primary_count=len(primary_docs),
                shadow_count=len(shadow_docs),
                primary_ms=primary_ms,
                shadow_ms=shadow_ms,
            )
            logger.info(
                "RAG dual-read parity: shadow_backend=%s overlap@k=%.3f primary=%d shadow=%d "
                "primary_ms=%.1f shadow_ms=%.1f",
                self._cfg.shadow_backend,
                metrics.overlap_at_k,
                metrics.primary_count,
                metrics.shadow_count,
                metrics.primary_ms,
                metrics.shadow_ms,
            )
            if self._cfg.log_payloads:
                logger.debug(
                    "RAG dual-read payloads: primary_keys=%s shadow_keys=%s",
                    [_doc_key(d) for d in primary_docs[:limit]],
                    [_doc_key(d) for d in shadow_docs[:limit]],
                )
        except asyncio.TimeoutError:
            logger.warning(
                "RAG dual-read shadow timed out: backend=%s timeout_ms=%d",
                self._cfg.shadow_backend,
                self._cfg.timeout_ms,
            )
        except Exception:
            logger.exception("RAG dual-read shadow failed")


__all__ = ["DualReadHarness", "ParityConfig"]
