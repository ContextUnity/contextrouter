"""Retrieval pipeline (pure orchestration).

- coordinates registered retrieval sources (providers + connectors)
- reranks, deduplicates, builds citations, attaches graph facts
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass
from typing import List

from contextrouter.core import (
    BiscuitToken,
    TokenBuilder,
    get_core_config,
)
from contextrouter.cortex import AgentState, get_graph_service, get_last_user_query
from contextrouter.modules.retrieval import BaseRetrievalPipeline

from .citations import build_citations
from .mmr import mmr_select
from .models import Citation, RetrievedDoc
from .parity import DualReadHarness, ParityConfig
from .rerankers import get_reranker
from .settings import RagRetrievalSettings, get_rag_retrieval_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Return value for `RetrievalPipeline.execute`."""

    retrieved_docs: list[RetrievedDoc]
    citations: list[Citation]
    graph_facts: list[str]


class RetrievalPipeline:
    """Orchestrates retrieval from multiple sources and builds citations."""

    def __init__(self) -> None:
        self.core_cfg = get_core_config()
        self._base = BaseRetrievalPipeline()

    def _token_from_state(self, state: AgentState) -> BiscuitToken:
        tok = state.get("access_token")
        if isinstance(tok, BiscuitToken):
            return tok

        if self.core_cfg.security.enabled:
            # If security is enabled, the runner MUST provide a valid token.
            # Minting a fresh one here would bypass authorization.
            raise PermissionError("Access token missing from AgentState in security-enabled mode")

        # Fallback for local development or unsecured deployments.
        builder = TokenBuilder(enabled=False)
        return builder.mint_root(
            user_ctx={},
            permissions=(self.core_cfg.security.policies.read_permission,),
            ttl_s=300.0,
        )

    async def execute(self, state: AgentState) -> RetrievalResult:
        pipeline_start = time.perf_counter()
        cfg = get_rag_retrieval_settings()
        user_query = (
            state.get("user_query") or get_last_user_query(state.get("messages", [])) or ""
        ).strip()
        if not user_query:
            return RetrievalResult(retrieved_docs=[], citations=[], graph_facts=[])

        retrieval_queries = self._normalize_queries(state, user_query)

        # Log taxonomy_concepts availability before graph facts lookup
        taxonomy_concepts = state.get("taxonomy_concepts") or []
        logger.debug(
            "RAG Pipeline: taxonomy_concepts in state: count=%d concepts=%s",
            len(taxonomy_concepts),
            taxonomy_concepts[:5] if taxonomy_concepts else [],
        )

        graph_facts = self._get_graph_facts(state)

        # Determine active provider
        active_providers = self._get_active_providers(cfg)
        provider_name = active_providers[0] if active_providers else "unknown"

        # Log retrieval start with provider info
        logger.debug(
            "RAG Retrieval Pipeline START: provider=%s query=%r queries=%d general_mode=%s",
            provider_name,
            user_query[:80],
            len(retrieval_queries),
            cfg.general_retrieval_enabled,
        )

        try:
            token = self._token_from_state(state)
        except Exception as e:
            from contextrouter.core.exceptions import RetrievalError

            raise RetrievalError(
                f"Failed to resolve access token: {str(e)}", code="AUTH_ERROR"
            ) from e

        all_docs: list[RetrievedDoc] = []

        # 1) Provider retrieval (storage boundary) via generic pipeline
        provider_start = time.perf_counter()
        try:
            provider_docs = await self._retrieve_from_providers(retrieval_queries, token, cfg)
            provider_elapsed_ms = (time.perf_counter() - provider_start) * 1000
            logger.debug(
                "Provider retrieval COMPLETE: provider=%s docs=%d elapsed_ms=%.1f",
                provider_name,
                len(provider_docs),
                provider_elapsed_ms,
            )
            all_docs.extend(provider_docs)
            self._run_dual_read(
                cfg=cfg,
                query=user_query,
                token=token,
                primary_docs=provider_docs,
                primary_elapsed_ms=provider_elapsed_ms,
            )
        except Exception as e:
            provider_elapsed_ms = (time.perf_counter() - provider_start) * 1000
            logger.error(
                "Critical provider retrieval failure: provider=%s elapsed_ms=%.1f error=%s",
                provider_name,
                provider_elapsed_ms,
                e,
            )

        # 2) Connector retrieval if enabled
        try:
            connector_docs = await self._retrieve_from_connectors(
                state, retrieval_queries, user_query, cfg
            )
            all_docs.extend(connector_docs)
        except Exception as e:
            logger.error("Critical connector retrieval failure: %s", e)

        if not all_docs:
            logger.warning(
                "No documents retrieved from any source for query: %r",
                user_query[:80],
            )

        # 3) Deduplicate
        deduped = self._deduplicate(all_docs)

        # 4) MMR (optional) + rerank and select
        reranker = get_reranker(cfg=cfg, provider=provider_name)
        if cfg.general_retrieval_enabled:
            candidates = deduped
            if cfg.mmr_enabled:
                candidates = mmr_select(
                    query=user_query,
                    candidates=candidates,
                    k=min(len(candidates), int(cfg.general_retrieval_initial_count)),
                    lambda_mult=float(cfg.mmr_lambda),
                )
            ranked_docs = await reranker.rerank(
                query=user_query,
                documents=candidates,
                top_n=cfg.general_retrieval_final_count,
            )
        else:
            candidates = deduped
            if cfg.mmr_enabled:
                total_limit = int(sum(self._type_limits(cfg).values()) or len(candidates))
                candidates = mmr_select(
                    query=user_query,
                    candidates=candidates,
                    k=min(len(candidates), total_limit),
                    lambda_mult=float(cfg.mmr_lambda),
                )
            ranked_all = await reranker.rerank(query=user_query, documents=candidates)
            ranked_docs = self._select_top_per_type(ranked_all, cfg)

        # 5) Citations (optional RAG capability)
        citations: list[Citation] = []
        if cfg.citations_enabled and (
            cfg.citations_books > 0
            or cfg.citations_videos > 0
            or cfg.citations_qa > 0
            or cfg.citations_web > 0
        ):
            citations = build_citations(
                ranked_docs,
                citations_books=cfg.citations_books,
                citations_videos=cfg.citations_videos,
                citations_qa=cfg.citations_qa,
                citations_web=cfg.citations_web,
            )
            if allowed_types := state.get("citations_allowed_types"):
                citations = [c for c in citations if c.source_type in allowed_types]

        pipeline_elapsed_ms = (time.perf_counter() - pipeline_start) * 1000
        logger.debug(
            "RAG Retrieval Pipeline COMPLETE: provider=%s total_docs=%d citations=%d graph_facts=%d elapsed_ms=%.1f",
            provider_name,
            len(ranked_docs),
            len(citations),
            len(graph_facts),
            pipeline_elapsed_ms,
        )

        return RetrievalResult(
            retrieved_docs=ranked_docs, citations=citations, graph_facts=graph_facts
        )

    def _type_limits(self, cfg: RagRetrievalSettings) -> dict[str, int]:
        """Resolve per-type limits for RAG retrieval.

        This is modular: new source types can be added via cfg.max_by_type.
        """
        if cfg.max_by_type:
            return {str(k): int(v) for k, v in cfg.max_by_type.items() if int(v) > 0}
        out = {
            "book": int(cfg.max_books),
            "video": int(cfg.max_videos),
            "qa": int(cfg.max_qa),
            "knowledge": int(cfg.max_knowledge),
        }
        return {k: v for k, v in out.items() if v > 0}

    def _coerce_doc_from_envelope(self, env: object) -> RetrievedDoc | None:
        """Convert BisquitEnvelope.content into RetrievedDoc when possible."""
        try:
            content = getattr(env, "content", None)
        except Exception:
            content = None
        if isinstance(content, RetrievedDoc):
            return content
        if isinstance(content, dict):
            try:
                return RetrievedDoc.model_validate(content)
            except Exception as e:
                logger.debug("Failed to validate RetrievedDoc from dict: %s", e)
        if isinstance(content, str) and content.strip():
            # Best-effort: unknown source type
            return RetrievedDoc(source_type="unknown", content=content)
        return None

    def _get_active_providers(self, cfg: RagRetrievalSettings) -> list[str]:
        """Get active providers list, preferring cfg.provider if set."""
        if cfg.provider and cfg.provider.strip():
            return [cfg.provider.strip()]
        return list(cfg.providers) if cfg.providers else ["vertex"]

    async def _retrieve_from_providers(
        self,
        retrieval_queries: list[str],
        token: BiscuitToken,
        cfg: RagRetrievalSettings,
    ) -> list[RetrievedDoc]:
        docs: list[RetrievedDoc] = []
        active_providers = self._get_active_providers(cfg)
        provider_name = active_providers[0] if active_providers else "unknown"

        logger.info(
            "Provider retrieval START: provider=%s queries=%d general_mode=%s",
            provider_name,
            len(retrieval_queries),
            cfg.general_retrieval_enabled,
        )

        if cfg.general_retrieval_enabled:
            limit = int(cfg.general_retrieval_initial_count)
            tasks = [
                self._base.execute(
                    q,
                    token=token,
                    limit=limit,
                    filters=None,
                    providers=active_providers,
                )
                for q in retrieval_queries
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error("Provider retrieval failed: %s", res)
                    continue
                for env in getattr(res, "envelopes", []) or []:
                    if d := self._coerce_doc_from_envelope(env):
                        docs.append(d)
            return docs

        # Per-type mode: run per source_type with its own limit
        limits = self._type_limits(cfg)
        calls = []
        for q in retrieval_queries:
            for st, lim in limits.items():
                calls.append(
                    self._base.execute(
                        q,
                        token=token,
                        limit=int(lim),
                        filters={"source_type": st},
                        providers=active_providers,
                    )
                )
        results = await asyncio.gather(*calls, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.error("Provider retrieval failed: %s", res)
                continue
            for env in getattr(res, "envelopes", []) or []:
                if d := self._coerce_doc_from_envelope(env):
                    docs.append(d)

        # Back-compat fallback: some datastores (or schemas) may not support filtering
        # by `source_type` at query-time. If per-type calls return nothing, retry once
        # without filters (behavior matches pre-split pipeline).
        if not docs:
            try:
                fallback_limit = int(getattr(cfg, "general_retrieval_initial_count", 30) or 30)
            except Exception:
                fallback_limit = 30
            # Make sure we fetch enough for downstream per-type selection.
            fallback_limit = max(fallback_limit, int(sum(limits.values()) or 15))
            tasks = [
                self._base.execute(
                    q,
                    token=token,
                    limit=fallback_limit,
                    filters=None,
                    providers=active_providers,
                )
                for q in retrieval_queries
            ]
            fallback_results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in fallback_results:
                if isinstance(res, Exception):
                    logger.error("Provider fallback retrieval failed: %s", res)
                    continue
                for env in getattr(res, "envelopes", []) or []:
                    if d := self._coerce_doc_from_envelope(env):
                        docs.append(d)
        return docs

    async def _retrieve_from_connectors(
        self,
        state: AgentState,
        retrieval_queries: list[str],
        user_query: str,
        cfg: RagRetrievalSettings,
    ) -> list[RetrievedDoc]:
        out: list[RetrievedDoc] = []
        for key in cfg.connectors:
            if not self._should_run_connector(key, state):
                continue
            if key == "web":
                out.extend(await self._retrieve_web(state, retrieval_queries, user_query))
                continue
            # Unknown connector keys are ignored unless user provides a custom connector with compatible init.
            logger.debug("Connector '%s' skipped (no built-in wiring)", key)
        return out

    def _should_run_connector(self, key: str, state: AgentState) -> bool:
        if key == "web":
            return self._should_run_web(state)
        return True

    async def _retrieve_web(
        self, state: AgentState, retrieval_queries: list[str], user_query: str
    ) -> list[RetrievedDoc]:
        allowed = state.get("web_allowed_domains", [])
        max_results = state.get("max_web_results", 10)

        from contextrouter.core.registry import ComponentFactory

        inst = ComponentFactory.create_connector(
            "web",
            query=retrieval_queries[0] if retrieval_queries else user_query,
            allowed_domains=list(allowed or []),
            max_results_per_domain=int(max_results),
            retrieval_queries=list(retrieval_queries),
        )

        out: list[RetrievedDoc] = []
        async for env in inst.connect():
            d = getattr(env, "content", None)
            if isinstance(d, RetrievedDoc):
                out.append(d)
        return out

    def _deduplicate(self, docs: List[RetrievedDoc]) -> List[RetrievedDoc]:
        seen: set[str] = set()
        out: List[RetrievedDoc] = []
        for d in docs:
            # Use hash for more efficient deduplication if content is large
            raw_key = f"{(d.url or '')}::{(d.snippet or '')}::{(d.content or '')}"
            key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out

    def _select_top_per_type(
        self, ranked: List[RetrievedDoc], cfg: RagRetrievalSettings
    ) -> List[RetrievedDoc]:
        limits = {
            "book": cfg.max_books,
            "video": cfg.max_videos,
            "qa": cfg.max_qa,
            "knowledge": cfg.max_knowledge,
        }
        buckets: dict[str, list[RetrievedDoc]] = {k: [] for k in limits}
        for d in ranked:
            st = str(getattr(d, "source_type", "") or "")
            if st in buckets and len(buckets[st]) < limits.get(st, 0):
                buckets[st].append(d)

        out: List[RetrievedDoc] = []
        for _, docs in buckets.items():
            out.extend(docs)
        return out

    def _normalize_queries(self, state: AgentState, user_query: str) -> List[str]:
        queries = state.get("retrieval_queries") or []
        out = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        if not out:
            out = [user_query]
        return out[:3]

    def _get_graph_facts(self, state: AgentState) -> List[str]:
        concepts = state.get("taxonomy_concepts") or []
        if not concepts:
            logger.debug(
                "Graph facts SKIPPED: taxonomy_concepts is empty or missing in state. "
                "Available state keys: %s",
                list(state.keys()) if isinstance(state, dict) else "not a dict",
            )
            return []

        logger.debug(
            "Graph facts lookup START: concepts=%d concepts_list=%s",
            len(concepts),
            concepts[:5] if concepts else [],
        )

        service = get_graph_service()
        try:
            facts = service.get_facts(concepts)[:50]
            logger.debug(
                "Graph facts lookup COMPLETE: concepts=%d facts=%d",
                len(concepts),
                len(facts),
            )
            return facts
        except Exception:
            logger.exception("Graph facts lookup failed: concepts=%s", concepts[:5])
            return []

    def _should_run_web(self, state: AgentState) -> bool:
        if state.get("enable_web_search") is False:
            return False
        domains = state.get("web_allowed_domains") or []
        return bool(domains)

    def _run_dual_read(
        self,
        *,
        cfg: RagRetrievalSettings,
        query: str,
        token: BiscuitToken,
        primary_docs: list[RetrievedDoc],
        primary_elapsed_ms: float,
    ) -> None:
        if not getattr(cfg, "dual_read_enabled", False):
            return
        parity_cfg = ParityConfig(
            enabled=bool(getattr(cfg, "dual_read_enabled", False)),
            shadow_backend=getattr(cfg, "dual_read_shadow_backend", None),
            sample_rate=float(getattr(cfg, "dual_read_sample_rate", 0.0) or 0.0),
            timeout_ms=int(getattr(cfg, "dual_read_timeout_ms", 300) or 300),
            log_payloads=bool(getattr(cfg, "dual_read_log_payloads", False)),
        )
        harness = DualReadHarness(parity_cfg)
        if not harness.should_run():
            return
        if cfg.general_retrieval_enabled:
            limit = int(cfg.general_retrieval_initial_count)
        else:
            limit = int(sum(self._type_limits(cfg).values()) or 15)
        asyncio.create_task(
            harness.compare(
                query=query,
                token=token,
                primary_docs=primary_docs,
                primary_ms=primary_elapsed_ms,
                limit=limit,
            )
        )


__all__ = ["RetrievalPipeline", "RetrievalResult"]
