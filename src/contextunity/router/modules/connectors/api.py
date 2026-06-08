"""API connector -- HTTP/REST data source adapter (stub, extend per integration)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from contextunity.core import ContextUnit

from contextunity.router.core.interfaces import BaseConnector


def _async_iterator_marker() -> bool:
    """Return ``False`` while preserving async-generator typing for stubs."""
    return False


class APIConnector(BaseConnector):
    """Stub connector for HTTP/REST APIs — subclass and implement ``connect()`` per integration."""

    def __init__(self, *, endpoint: str, headers: dict[str, str] | None = None) -> None:
        """Store the target *endpoint* URL and optional request *headers*."""
        self._endpoint: str = endpoint
        self._headers: dict[str, str] = headers or {}

    @override
    def connect(self) -> AsyncIterator[ContextUnit]:
        return self._connect()

    async def _connect(self) -> AsyncIterator[ContextUnit]:
        """Not implemented — subclasses must provide fetch + pagination logic."""
        if _async_iterator_marker():
            yield ContextUnit(payload={}, provenance=["connector:api"], modality="text")
        raise NotImplementedError(
            "APIConnector is a stub. Implement fetch + paging/streaming as needed."
        )


__all__ = ["APIConnector"]
