"""AG-UI tool event definitions.
These events follow the AG-UI protocol specification for tool calls.
See: https://docs.ag-ui.com/llms-full.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NotRequired, TypedDict

from contextunity.core.types import JsonDict, JsonValue


class AguiEventDict(TypedDict, total=False):
    """Wire-format envelope for AG-UI tool-call events (serialized to JSON)."""

    type: str
    toolCallId: str
    timestamp: float
    toolName: NotRequired[str]
    args: NotRequired[JsonDict]
    result: NotRequired[JsonValue]
    # ContextUnit extension fields (used by contextunit_mapper)
    tokenId: NotRequired[str | None]
    provenance: NotRequired[list[str]]
    citations: NotRequired[list[JsonDict]]
    metadata: NotRequired[JsonDict]
    data: NotRequired[JsonValue]


@dataclass
class ToolCallStart:
    """Emitted when the agent begins a tool invocation."""

    toolCallId: str
    name: str
    timestamp: float

    def to_dict(self) -> AguiEventDict:
        """Serialize to an ``AguiEventDict`` wire-format envelope."""
        return {
            "type": "ToolCallStart",
            "toolCallId": self.toolCallId,
            "toolName": self.name,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallArgs:
    """Carries the resolved arguments for a tool invocation."""

    toolCallId: str
    args: JsonDict
    timestamp: float

    def to_dict(self) -> AguiEventDict:
        """Serialize to an ``AguiEventDict`` wire-format envelope."""
        return {
            "type": "ToolCallArgs",
            "toolCallId": self.toolCallId,
            "args": self.args,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallEnd:
    """Marks the end of a tool invocation (before result is available)."""

    toolCallId: str
    timestamp: float

    def to_dict(self) -> AguiEventDict:
        """Serialize to an ``AguiEventDict`` wire-format envelope."""
        return {
            "type": "ToolCallEnd",
            "toolCallId": self.toolCallId,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallResult:
    """Carries the return value of a completed tool invocation."""

    toolCallId: str
    result: JsonValue
    timestamp: float

    def to_dict(self) -> AguiEventDict:
        """Serialize to an ``AguiEventDict`` wire-format envelope."""
        return {
            "type": "ToolCallResult",
            "toolCallId": self.toolCallId,
            "result": self.result,
            "timestamp": self.timestamp,
        }


__all__ = ["ToolCallStart", "ToolCallArgs", "ToolCallEnd", "ToolCallResult"]
