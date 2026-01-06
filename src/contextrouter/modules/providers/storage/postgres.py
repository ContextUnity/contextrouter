"""Postgres provider (storage sink) - placeholder."""

from __future__ import annotations

from typing import Any

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseProvider, IRead, IWrite
from contextrouter.core.tokens import AccessManager, BiscuitToken


class PostgresProvider(BaseProvider, IRead, IWrite):
    async def read(
        self,
        query: str,
        *,
        limit: int = 5,
        filters: dict[str, Any] | None = None,
        token: BiscuitToken,
    ) -> list[BisquitEnvelope]:
        _ = query, limit, filters, token
        raise NotImplementedError("PostgresProvider.read is not implemented yet")

    async def write(self, data: BisquitEnvelope, *, token: BiscuitToken) -> None:
        AccessManager.from_core_config().verify_envelope_write(data, token)
        _ = data, token
        raise NotImplementedError("PostgresProvider.write is not implemented yet")

    async def sink(self, envelope: BisquitEnvelope, *, token: BiscuitToken) -> Any:
        await self.write(envelope, token=token)
        return None


__all__ = ["PostgresProvider"]
