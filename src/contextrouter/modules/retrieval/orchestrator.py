"""RetrievalOrchestrator: coordinate search across registered providers.

This is a thin orchestration layer (deep modules stay in providers/storage/retrieval).
It coordinates `IRead` implementations registered in `provider_registry`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from contextrouter.core import (
    AccessManager,
    BiscuitToken,
    BisquitEnvelope,
    IRead,
)
from contextrouter.core.registry import ComponentFactory
from contextrouter.core.types import QueryLike, normalize_query


@dataclass(frozen=True)
class RetrievalResult:
    envelopes: list[BisquitEnvelope]


class RetrievalOrchestrator:
    """Fan-out retrieval across IRead providers and merge results."""

    def __init__(self, *, access: AccessManager | None = None) -> None:
        self._access = access or AccessManager.from_core_config()

    async def search(
        self,
        query: QueryLike,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: BiscuitToken,
        providers: list[str] | None = None,
    ) -> RetrievalResult:
        self._access.verify_read(token)

        query_text, extra = normalize_query(query)
        merged_filters: dict[str, Any] | None
        if filters is None and extra is None:
            merged_filters = None
        else:
            merged_filters = dict(filters or {})
            if isinstance(extra, dict):
                merged_filters.update(extra)

        keys = providers or ["vertex"]  # Default to vertex provider
        calls: list[tuple[str, object]] = []
        for key in keys:
            try:
                inst = ComponentFactory.create_provider(key)
            except ValueError:
                # Skip unknown providers
                continue
            if isinstance(inst, IRead):
                calls.append(
                    (
                        key,
                        inst.read(query_text, limit=limit, filters=merged_filters, token=token),
                    )
                )

        if not calls:
            return RetrievalResult(envelopes=[])

        results = await asyncio.gather(*(coro for _, coro in calls), return_exceptions=True)
        merged: list[BisquitEnvelope] = []
        for (key, _), r in zip(calls, results):
            if isinstance(r, Exception):
                # Do not silently swallow provider failures; they explain "0 docs" cases.
                # Keep this at warning (not error) because we still attempt other sources.
                # Provide a compact message to avoid dumping secrets.
                # Note: provider stack traces are logged at the provider boundary.
                # Here we keep the orchestrator message short and actionable.
                # Example: Vertex config/serving_config/resource errors.
                # pylint/ruff: ignore nosec - message is controlled.
                # type: ignore[reportGeneralTypeIssues]
                import logging

                logging.getLogger(__name__).warning("Provider '%s' failed: %s", key, r)
                continue
            if isinstance(r, list):
                merged.extend([x for x in r if isinstance(x, BisquitEnvelope)])
        return RetrievalResult(envelopes=merged)


__all__ = ["RetrievalOrchestrator", "RetrievalResult"]
