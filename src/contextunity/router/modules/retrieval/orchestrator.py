"""RetrievalOrchestrator: coordinate search across registered providers.
This is a thin orchestration layer (deep modules stay in providers/storage/retrieval).
It coordinates `IRead` implementations registered in `provider_registry`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TypeGuard

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.types import JsonDict, is_object_list

from contextunity.router.core import (
    IRead,
)
from contextunity.router.core.registry import ComponentFactory
from contextunity.router.core.types import QueryLike, normalize_query


def _is_context_unit_list(value: object) -> TypeGuard[list[ContextUnit]]:
    if not is_object_list(value):
        return False
    return all(isinstance(item, ContextUnit) for item in value)


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable result of a fan-out retrieval across multiple IRead providers."""

    units: list[ContextUnit]


class RetrievalOrchestrator:
    """Fan-out retrieval across IRead providers and merge results."""

    def __init__(self) -> None:
        """No-op — the orchestrator is stateless; providers are resolved at search time."""
        pass

    async def search(
        self,
        query: QueryLike,
        *,
        limit: int = 5,
        filters: JsonDict | None = None,
        providers: list[str] | None = None,
    ) -> RetrievalResult:
        """Search."""
        query_text, extra = normalize_query(query)
        merged_filters: JsonDict | None
        if filters is None and extra is None:
            merged_filters = None
        else:
            merged_filters = dict(filters or {})
            if extra is not None:
                merged_filters.update(extra)

        keys = providers or ["vertex"]  # Default to vertex provider
        calls: list[tuple[str, Awaitable[list[ContextUnit]]]] = []
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
                        inst.read(query_text, limit=limit, filters=merged_filters),
                    )
                )

        if not calls:
            return RetrievalResult(units=[])

        results = await asyncio.gather(*(coro for _, coro in calls), return_exceptions=True)
        merged: list[ContextUnit] = []
        for (key, _), r in zip(calls, results):
            if isinstance(r, Exception):
                # Do not silently swallow provider failures; they explain "0 docs" cases.
                # Keep this at warning (not error) because we still attempt other sources.
                # Provide a compact message to avoid dumping secrets.
                # Note: provider stack traces are logged at the provider boundary.
                # Here we keep the orchestrator message short and actionable.
                # Example: Vertex config/serving_config/resource errors.
                # pylint/ruff: ignore nosec - message is controlled.

                get_contextunit_logger(__name__).warning("Provider '%s' failed: %s", key, r)
                continue
            if _is_context_unit_list(r):
                merged.extend(r)
        return RetrievalResult(units=merged)


__all__ = ["RetrievalOrchestrator", "RetrievalResult"]
