"""API connector (source) - stub."""

from __future__ import annotations

from typing import AsyncIterator

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseConnector


class APIConnector(BaseConnector):
    def __init__(self, *, endpoint: str, headers: dict[str, str] | None = None) -> None:
        self._endpoint = endpoint
        self._headers = headers or {}

    async def connect(self) -> AsyncIterator[BisquitEnvelope]:
        raise NotImplementedError(
            "APIConnector is a stub. Implement fetch + paging/streaming as needed."
        )


__all__ = ["APIConnector"]
