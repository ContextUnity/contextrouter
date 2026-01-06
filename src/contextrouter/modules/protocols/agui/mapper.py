"""AG-UI event mapper.

Maps internal tool events to AG-UI protocol events.
"""

from __future__ import annotations

import time
from typing import Any

from .events import (
    ToolCallArgs,
    ToolCallEnd,
    ToolCallResult,
    ToolCallStart,
)


class AguiMapper:
    """Maps tool events to AG-UI protocol events."""

    def __init__(self) -> None:
        self._tool_call_ids: dict[str, str] = {}
        self._call_counter = 0

    def _tool_key(self, *, tool_name: str, tool_key: str | None) -> str:
        if tool_key:
            return f"{tool_key}:{tool_name}"
        return tool_name

    def map_tool_start(
        self,
        tool_name: str,
        args: dict[str, Any] | None = None,
        *,
        tool_key: str | None = None,
    ) -> ToolCallStart:
        _ = args
        self._call_counter += 1
        tool_call_id = f"call-{int(time.time() * 1000)}-{self._call_counter}-{tool_name}"
        key = self._tool_key(tool_name=tool_name, tool_key=tool_key)
        self._tool_call_ids[key] = tool_call_id

        return ToolCallStart(toolCallId=tool_call_id, name=tool_name, timestamp=time.time())

    def map_tool_args(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        tool_key: str | None = None,
    ) -> ToolCallArgs:
        key = self._tool_key(tool_name=tool_name, tool_key=tool_key)
        tool_call_id = self._tool_call_ids.get(key)
        if not tool_call_id:
            raise ValueError(f"No tool_call_id for {tool_name}")
        return ToolCallArgs(toolCallId=tool_call_id, args=args, timestamp=time.time())

    def map_tool_end(self, tool_name: str, *, tool_key: str | None = None) -> ToolCallEnd:
        key = self._tool_key(tool_name=tool_name, tool_key=tool_key)
        tool_call_id = self._tool_call_ids.get(key)
        if not tool_call_id:
            raise ValueError(f"No tool_call_id for {tool_name}")
        return ToolCallEnd(toolCallId=tool_call_id, timestamp=time.time())

    def map_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        tool_key: str | None = None,
    ) -> ToolCallResult:
        key = self._tool_key(tool_name=tool_name, tool_key=tool_key)
        tool_call_id = self._tool_call_ids.get(key)
        if not tool_call_id:
            raise ValueError(f"No tool_call_id for {tool_name}")
        return ToolCallResult(toolCallId=tool_call_id, result=result, timestamp=time.time())

    def clear_tool_call(self, tool_name: str, *, tool_key: str | None = None) -> None:
        key = self._tool_key(tool_name=tool_name, tool_key=tool_key)
        self._tool_call_ids.pop(key, None)


__all__ = ["AguiMapper"]
