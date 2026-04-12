"""Connector tool wrappers.

These wrappers allow agents to treat connectors as standard callable tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from contextunity.core import ContextUnit

from contextunity.router.core.interfaces import BaseConnector


@dataclass(frozen=True)
class ConnectorTool:
    """Minimal connector-as-tool wrapper."""

    name: str
    connector: BaseConnector

    async def run(self) -> list[ContextUnit]:
        out: list[ContextUnit] = []
        async for env in self.connector.connect():
            out.append(env)
        return out


__all__ = ["ConnectorTool"]
