"""RSS connector (source) - stub."""

from __future__ import annotations

from typing import AsyncIterator

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseConnector


class RSSConnector(BaseConnector):
    def __init__(self, *, feed_url: str) -> None:
        self._feed_url = feed_url

    async def connect(self) -> AsyncIterator[BisquitEnvelope]:
        raise NotImplementedError(
            "RSSConnector is a stub. Implement feed parsing and item extraction."
        )


__all__ = ["RSSConnector"]
