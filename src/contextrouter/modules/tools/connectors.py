"""Connector tool wrappers.

These wrappers allow agents to treat connectors as standard callable tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.interfaces import BaseConnector


@dataclass(frozen=True)
class ConnectorTool:
    """Minimal connector-as-tool wrapper."""

    name: str
    connector: BaseConnector

    async def run(self) -> list[BisquitEnvelope]:
        out: list[BisquitEnvelope] = []
        async for env in self.connector.connect():
            out.append(env)
        return out


__all__ = ["ConnectorTool"]
