"""Pipeline retrieval methods - providers, connectors, web."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from contextrouter.core import ContextToken
from contextrouter.cortex.state import AgentState
from contextrouter.modules.retrieval import BaseRetrievalPipeline

from .models import RetrievedDoc
from .parity import DualReadHarness, ParityConfig
from .pipeline_helpers import coerce_doc_from_envelope, get_type_limits
from .settings import RagRetrievalSettings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RetrievalMixin:
    """Mixin providing retrieval methods for RetrievalPipeline."""

    _base: BaseRetrievalPipeline

    def _get_active_providers(self, cfg: RagRetrievalSettings) -> list[str]:
        """Get active providers list, preferring cfg.provider if set."""
        if cfg.provider and cfg.provider.strip():
            return [cfg.provider.strip()]
        return list(cfg.providers) if cfg.providers else ["brain"]

    async def _retrieve_from_providers(
        self,
        retrieval_queries: list[str],
        token: ContextToken,
        cfg: RagRetrievalSettings,
    ) -> list[RetrievedDoc]:
        """Retrieve documents from configured providers."""
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
                for env in getattr(res, "units", []) or []:
                    if d := coerce_doc_from_envelope(env):
                        docs.append(d)
            return docs

        # Per-type mode
        limits = get_type_limits(cfg)
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
            for env in getattr(res, "units", []) or []:
                if d := coerce_doc_from_envelope(env):
                    docs.append(d)

        # Fallback if per-type returns nothing
        if not docs:
            try:
                fallback_limit = int(getattr(cfg, "general_retrieval_initial_count", 30) or 30)
            except Exception:
                fallback_limit = 30
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
                for env in getattr(res, "units", []) or []:
                    if d := coerce_doc_from_envelope(env):
                        docs.append(d)
        return docs

    async def _retrieve_from_connectors(
        self,
        state: AgentState,
        retrieval_queries: list[str],
        user_query: str,
        cfg: RagRetrievalSettings,
    ) -> list[RetrievedDoc]:
        """Retrieve documents from connectors (web, etc.)."""
        out: list[RetrievedDoc] = []
        for key in cfg.connectors:
            if not self._should_run_connector(key, state):
                continue
            if key == "web":
                out.extend(await self._retrieve_web(state, retrieval_queries, user_query))
                continue
            logger.debug("Connector '%s' skipped (no built-in wiring)", key)
        return out

    def _should_run_connector(self, key: str, state: AgentState) -> bool:
        """Check if a connector should run."""
        if key == "web":
            return self._should_run_web(state)
        return True

    def _should_run_web(self, state: AgentState) -> bool:
        """Check if web search should run."""
        if state.get("enable_web_search") is False:
            return False
        domains = state.get("web_allowed_domains") or []
        return bool(domains)

    async def _retrieve_web(
        self, state: AgentState, retrieval_queries: list[str], user_query: str
    ) -> list[RetrievedDoc]:
        """Retrieve documents from web search."""
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

    def _run_dual_read(
        self,
        *,
        cfg: RagRetrievalSettings,
        query: str,
        token: ContextToken,
        primary_docs: list[RetrievedDoc],
        primary_elapsed_ms: float,
    ) -> None:
        """Run dual-read comparison if enabled."""
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
            limit = int(sum(get_type_limits(cfg).values()) or 15)
        asyncio.create_task(
            harness.compare(
                query=query,
                token=token,
                primary_docs=primary_docs,
                primary_ms=primary_elapsed_ms,
                limit=limit,
            )
        )


__all__ = ["RetrievalMixin"]
