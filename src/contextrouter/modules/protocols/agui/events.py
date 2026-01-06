"""AG-UI tool event definitions.

These events follow the AG-UI protocol specification for tool calls.
See: https://docs.ag-ui.com/llms-full.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCallStart:
    toolCallId: str
    name: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ToolCallStart",
            "toolCallId": self.toolCallId,
            "toolName": self.name,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallArgs:
    toolCallId: str
    args: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ToolCallArgs",
            "toolCallId": self.toolCallId,
            "args": self.args,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallEnd:
    toolCallId: str
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ToolCallEnd",
            "toolCallId": self.toolCallId,
            "timestamp": self.timestamp,
        }


@dataclass
class ToolCallResult:
    toolCallId: str
    result: Any
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ToolCallResult",
            "toolCallId": self.toolCallId,
            "result": self.result,
            "timestamp": self.timestamp,
        }


__all__ = ["ToolCallStart", "ToolCallArgs", "ToolCallEnd", "ToolCallResult"]
