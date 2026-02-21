"""Retrieval pipeline (pure orchestration).

- coordinates registered retrieval sources (providers + connectors)
- reranks, deduplicates, builds citations, attaches graph facts
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from contextrouter.core import (
    ContextToken,
    TokenBuilder,
    get_core_config,
)
from contextrouter.cortex.state import AgentState, get_last_user_query
from contextrouter.modules.retrieval import BaseRetrievalPipeline

from .citations import build_citations
from .mmr import mmr_select
from .models import Citation, RetrievedDoc
from .pipeline_helpers import (
    deduplicate_docs,
    get_graph_facts,
    get_type_limits,
    normalize_queries,
    select_top_per_type,
)
from .pipeline_retrieval import RetrievalMixin
from .rerankers import get_reranker
from .settings import get_rag_retrieval_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Return value for `RetrievalPipeline.execute`."""

    retrieved_docs: list[RetrievedDoc]
    citations: list[Citation]
    graph_facts: list[str]


class RetrievalPipeline(RetrievalMixin):
    """Orchestrates retrieval from multiple sources and builds citations."""

    def __init__(self) -> None:
        self.core_cfg = get_core_config()
        self._base = BaseRetrievalPipeline()

    def _token_from_state(self, state: AgentState) -> ContextToken:
        """Resolve access token from agent state."""
        tok = state.get("access_token")
        if isinstance(tok, ContextToken):
            return tok

        if self.core_cfg.security.enabled:
            raise PermissionError("Access token missing from AgentState in security-enabled mode")

        builder = TokenBuilder(enabled=False)
        return builder.mint_root(
            user_ctx={},
            permissions=(self.core_cfg.security.policies.read_permission,),
            ttl_s=300.0,
        )

    async def execute(self, state: AgentState) -> RetrievalResult:
        """Execute retrieval pipeline and return results."""
        pipeline_start = time.perf_counter()
        cfg = get_rag_retrieval_settings()
        user_query = (
            state.get("user_query") or get_last_user_query(state.get("messages", [])) or ""
        ).strip()
        if not user_query:
            return RetrievalResult(retrieved_docs=[], citations=[], graph_facts=[])

        retrieval_queries = normalize_queries(state, user_query)

        # Log taxonomy_concepts availability
        taxonomy_concepts = state.get("taxonomy_concepts") or []
        logger.debug(
            "RAG Pipeline: taxonomy_concepts in state: count=%d concepts=%s",
            len(taxonomy_concepts),
            taxonomy_concepts[:5] if taxonomy_concepts else [],
        )

        graph_facts = get_graph_facts(state)

        # Determine active provider
        active_providers = self._get_active_providers(cfg)
        provider_name = active_providers[0] if active_providers else "unknown"

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
            from contextcore.exceptions import RetrievalError

            raise RetrievalError(
                f"Failed to resolve access token: {str(e)}", code="AUTH_ERROR"
            ) from e

        all_docs: list[RetrievedDoc] = []

        # 1) Provider retrieval
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

        # 2) Connector retrieval
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
        deduped = deduplicate_docs(all_docs)

        # 4) MMR + rerank
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
                total_limit = int(sum(get_type_limits(cfg).values()) or len(candidates))
                candidates = mmr_select(
                    query=user_query,
                    candidates=candidates,
                    k=min(len(candidates), total_limit),
                    lambda_mult=float(cfg.mmr_lambda),
                )
            ranked_all = await reranker.rerank(query=user_query, documents=candidates)
            ranked_docs = select_top_per_type(ranked_all, cfg)

        # 5) Citations
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


__all__ = ["RetrievalPipeline", "RetrievalResult"]
