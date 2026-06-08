"""JSON-like wire values for tool results, AG-UI payloads, and tracing.

Recursive alias avoids ``Any`` while matching JSON-serializable shapes used at
LangGraph / SSE boundaries (see type-validation skill).
"""

from __future__ import annotations

from typing import TypeAlias

JsonWireValue: TypeAlias = (
    str | int | float | bool | None | list["JsonWireValue"] | dict[str, "JsonWireValue"]
)

__all__ = ["JsonWireValue"]
