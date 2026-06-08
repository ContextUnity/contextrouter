"""RSS connector -- feed polling and article extraction adapter (stub, extend per integration)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from contextunity.core import ContextUnit

from contextunity.router.core.interfaces import BaseConnector


def _async_iterator_marker() -> bool:
    """Return ``False`` while preserving async-generator typing for stubs."""
    return False


class RSSConnector(BaseConnector):
    """Stub connector for RSS/Atom feeds — subclass and implement ``connect()`` to parse entries."""

    def __init__(self, *, feed_url: str) -> None:
        """Store the target *feed_url*."""
        self._feed_url: str = feed_url

    @override
    def connect(self) -> AsyncIterator[ContextUnit]:
        return self._connect()

    async def _connect(self) -> AsyncIterator[ContextUnit]:
        """Not implemented — subclasses must provide feed parsing and item extraction logic."""
        if _async_iterator_marker():
            yield ContextUnit(payload={}, provenance=["connector:rss"], modality="text")
        raise NotImplementedError(
            "RSSConnector is a stub. Implement feed parsing and item extraction."
        )


__all__ = ["RSSConnector"]
