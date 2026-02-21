"""Shared helpers for SQL analytics graph nodes."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from contextrouter.core import get_core_config
from contextrouter.modules.models.types import ModelRequest, TextPart

logger = logging.getLogger(__name__)

# Verbose graph-level logging — lazy to avoid import-time config access
_dbg_cache: bool | None = None


def is_debug() -> bool:
    """Check if graph-level debug logging is enabled."""
    global _dbg_cache
    if _dbg_cache is None:
        try:
            _dbg_cache = get_core_config().debug_graph_messages
        except Exception:
            _dbg_cache = False
    return _dbg_cache


def extract_json(text: str) -> dict:
    """Robust JSON extraction from LLM response text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue

    # Last resort: bracket-counting extraction
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {}


async def invoke_model(
    model: Any,
    messages: list[BaseMessage],
) -> tuple[AIMessage, dict]:
    """Bridge LangChain messages to Router model.generate() API.

    Extracts system prompt from SystemMessage, concatenates remaining
    messages into a single TextPart, and returns result as AIMessage.
    """
    system = None
    parts_text: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system = msg.content
        elif isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            parts_text.append(f"{role}: {msg.content}")
        else:
            parts_text.append(str(msg.content))

    request = ModelRequest(
        parts=[TextPart(text="\n\n".join(parts_text))],
        system=system,
        response_format="json_object",
    )
    response = await model.generate(request)
    usage = response.usage
    usage_dict = {
        "input_tokens": usage.input_tokens or 0 if usage else 0,
        "output_tokens": usage.output_tokens or 0 if usage else 0,
        "total_cost": usage.total_cost or 0.0 if usage else 0.0,
    }
    return AIMessage(content=response.text), usage_dict


def acc_tokens(state: dict, usage: dict) -> dict:
    """Merge new usage into accumulated _token_usage."""
    prev = state.get("_token_usage") or {}
    return {
        "input_tokens": prev.get("input_tokens", 0) + usage.get("input_tokens", 0),
        "output_tokens": prev.get("output_tokens", 0) + usage.get("output_tokens", 0),
        "total_cost": prev.get("total_cost", 0.0) + usage.get("total_cost", 0.0),
    }


def validate_sql_syntax(sql: str) -> str | None:
    """Quick pre-validation for obvious SQL syntax errors.

    Returns error message if invalid, None if OK.
    """
    if re.search(r"\bIN\s*\(\s*\)", sql, re.IGNORECASE):
        return "SQL contains empty IN() clause. Use a valid list or remove the condition."

    if sql.count("(") != sql.count(")"):
        return f"Unbalanced parentheses: {sql.count('(')} opening vs {sql.count(')')} closing."

    return None


class StepTimer:
    """Context manager that measures wall-clock time for a graph node.

    Usage::

        timer = StepTimer()
        with timer:
            result = await do_work()
        return {"_steps": [step("planner", timer=timer, request={...}, result={...})]}
    """

    def __init__(self) -> None:
        import time

        self._start: float = time.monotonic()
        self.elapsed_ms: int = 0

    def __enter__(self) -> "StepTimer":
        import time

        self._start = time.monotonic()
        return self

    def __exit__(self, *_: Any) -> None:
        import time

        self.elapsed_ms = int((time.monotonic() - self._start) * 1000)


def step(
    name: str,
    *,
    status: str = "ok",
    timer: StepTimer | None = None,
    request: dict[str, Any] | str | None = None,
    result: dict[str, Any] | str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create a rich trace step record for the _steps accumulator.

    The record format is compatible with ContextView's ``metadata.steps``
    rendering (Graph Journey, Conversation Flow, per-tool timing).

    Args:
        name: Node/tool name (e.g. ``"planner"``, ``"execute_sql"``).
        status: ``"ok"`` | ``"error"`` | ``"skipped"``.
        timer: Optional :class:`StepTimer` — adds ``timing_ms``.
        request: Optional request/input data (truncated to 3 000 chars).
        result: Optional result/output data (truncated to 10 000 chars).
        **extra: Additional arbitrary metadata (e.g. ``row_count=42``).

    Usage in a node::

        timer = StepTimer()
        with timer:
            response, usage = await invoke_model(llm, msgs)
        return {
            "_steps": [step("planner", timer=timer,
                            request={"question": q}, result={"sql": sql})],
        }
    """
    import time

    record: dict[str, Any] = {
        "tool": name,
        "status": status,
        "ts": time.time(),
    }
    if timer is not None:
        record["timing_ms"] = timer.elapsed_ms
    if request is not None:
        req_s = str(request) if not isinstance(request, str) else request
        record["request"] = req_s[:3000]
    if result is not None:
        res_s = str(result) if not isinstance(result, str) else result
        record["result"] = res_s[:10000]
    record.update(extra)
    return record


__all__ = [
    "StepTimer",
    "acc_tokens",
    "extract_json",
    "invoke_model",
    "is_debug",
    "step",
    "validate_sql_syntax",
]
