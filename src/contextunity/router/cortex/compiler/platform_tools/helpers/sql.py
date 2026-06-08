"""Shared helpers and schemas for SQL analytics graph nodes."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TypedDict

from contextunity.core import get_contextunit_logger
from contextunity.core.parsing import json_dumps, json_loads
from contextunity.core.types import is_object_dict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

from contextunity.router.core import get_core_config
from contextunity.router.cortex.types import GraphState, extract_message_content
from contextunity.router.modules.models.base import BaseLLM as LLMBaseModel
from contextunity.router.modules.models.types import ModelRequest, TextPart

logger = get_contextunit_logger(__name__)


JsonMap = dict[str, object]


# ── SQL schemas ──────────────────────────────────────────────────────


class SqlResultDict(TypedDict, total=False):
    """Result of executing SQL query."""

    rows: list[JsonMap]
    columns: list[str]
    row_count: int
    duration_ms: float
    error: str


class ValidationDict(TypedDict, total=False):
    """Result of SQL data validation."""

    valid: bool
    reason: str
    hints: list[str]
    issues: list[str]
    warning: str


class TokenUsageDict(TypedDict, total=False):
    """Accumulated token usage tracking."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float


# ── Helpers ──────────────────────────────────────────────────────────

# Verbose graph-level logging — lazy to avoid import-time config access
_dbg_cache: bool | None = None


def is_debug() -> bool:
    """Check if graph-level debug logging is enabled."""
    global _dbg_cache
    if _dbg_cache is None:
        try:
            _dbg_cache = get_core_config().debug_graph_messages
        except Exception:  # graceful-degrade: SQL error returns empty result
            _dbg_cache = False
    return _dbg_cache


def extract_json(text: str) -> JsonMap:
    """Robust JSON extraction from LLM response text."""
    text = text.strip()
    try:
        payload = json_loads(text)
        if is_object_dict(payload):
            return {str(key): value for key, value in payload.items()}
    except Exception:
        pass

    # Try extracting from markdown code block
    for pattern in (r"```json\s*([\s\S]*?)```", r"```\s*([\s\S]*?)```"):
        m = re.search(pattern, text)
        if m:
            try:
                payload = json_loads(m.group(1).strip())
                if is_object_dict(payload):
                    return {str(key): value for key, value in payload.items()}
            except Exception:
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
                        payload = json_loads(text[start : i + 1])
                        if is_object_dict(payload):
                            return {str(key): value for key, value in payload.items()}
                    except Exception:
                        break
    return {}


async def invoke_model(
    model: LLMBaseModel,
    messages: Sequence[BaseMessage],
    *,
    config: RunnableConfig | None = None,
    prompt_version: str | None = None,
    node_name: str | None = None,
    state: GraphState | None = None,
) -> tuple[AIMessage, TokenUsageDict]:
    """Bridge LangChain messages to Router model.generate() API.

    Token is resolved from the context var set by ``secure_node``.
    Extracts system prompt from SystemMessage, concatenates remaining
    messages into a single TextPart, and returns result as AIMessage.

    All callback tracing (model name, prompt_version, tokens, cost)
    is handled by :func:`model_telemetry` — the single entry point
    for traced LLM invocations.

    If *prompt_version* is ``None`` and *node_name* + *state* are
    provided, prompt_version is resolved automatically from the
    manifest ``project_config`` in state.
    """
    system_parts: list[str] = []
    parts_text: list[str] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_parts.append(extract_message_content(msg).strip())
        elif isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            parts_text.append(f"{role}: {extract_message_content(msg)}")
        else:
            parts_text.append(extract_message_content(msg))

    system = "\n\n".join(system_parts) if system_parts else None

    user_text = "\n\n".join(parts_text)
    logger.info(
        "invoke_model: system_len=%s, user_text_len=%d",
        len(system) if system else 0,
        len(user_text),
    )

    request = ModelRequest(
        parts=[TextPart(text=user_text)],
        system=system,
        response_format="json_object",
    )
    logger.info("invoke_model: request.system len=%s", len(request.system) if request.system else 0)

    # ── Traced LLM call ──────────────────────────────────────────────
    # Delegate all callback tracing to the centralized utility.
    # Pass original LangChain messages for rich trace display.
    from contextunity.router.cortex.compiler.node_executors.telemetry import (
        model_telemetry,
    )

    response = await model_telemetry(
        model,
        request,
        config,
        prompt_version=prompt_version,
        node_name=node_name,
        state=state,
        trace_messages=list(messages),
    )

    usage = response.usage
    usage_dict: TokenUsageDict
    if usage:
        usage_dict = {
            "input_tokens": usage.input_tokens or 0,
            "output_tokens": usage.output_tokens or 0,
            "total_tokens": (usage.input_tokens or 0) + (usage.output_tokens or 0),
            "total_cost": usage.total_cost or 0.0,
        }
    else:
        usage_dict = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
        }

    return AIMessage(
        content=response.text,
        usage_metadata={
            "input_tokens": usage_dict["input_tokens"],
            "output_tokens": usage_dict["output_tokens"],
            "total_tokens": usage_dict["input_tokens"] + usage_dict["output_tokens"],
        },
        response_metadata={
            "model_name": response.raw_provider.model_name
            if getattr(response, "raw_provider", None)
            else "unknown",
            "total_cost": usage_dict["total_cost"],
        },
    ), usage_dict


def acc_tokens(state: GraphState, usage: TokenUsageDict) -> TokenUsageDict:
    """Merge new usage into accumulated _token_usage."""
    prev_raw: object = state.get("_token_usage")
    prev = prev_raw if is_object_dict(prev_raw) else {}
    prev_input = prev.get("input_tokens", 0)
    prev_output = prev.get("output_tokens", 0)
    prev_total_cost = prev.get("total_cost", 0.0)
    in_tok = (prev_input if isinstance(prev_input, int) else 0) + usage.get("input_tokens", 0)
    out_tok = (prev_output if isinstance(prev_output, int) else 0) + usage.get("output_tokens", 0)
    total_cost = (
        prev_total_cost if isinstance(prev_total_cost, (int, float)) else 0.0
    ) + usage.get("total_cost", 0.0)
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "total_cost": total_cost,
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


# ── SQL data analysis utilities ──────────────────────────────────────


def needs_chart(rows: list[JsonMap], columns: list[str]) -> bool:
    """Charts are useful when there are numeric columns and >1 row."""
    if len(rows) < 2:
        return False
    numeric_count = 0
    for col in columns:
        for r in rows[:5]:
            v = r.get(col)
            if isinstance(v, (int, float)):
                numeric_count += 1
                break
            if v is not None:
                try:
                    _ = float(str(v))
                    numeric_count += 1
                    break
                except (ValueError, TypeError):
                    pass
    return numeric_count >= 1


def needs_table(rows: list[JsonMap]) -> bool:
    """Tables are useful if there's data."""
    return len(rows) > 0


def compute_column_summary(rows: list[JsonMap], columns: list[str]) -> dict[str, dict[str, float]]:
    """Compute summary stats (count, sum, min, max, avg) for numeric columns."""
    summary: dict[str, dict[str, float]] = {}
    if not rows or not columns:
        return summary
    for col in columns:
        nums: list[float] = []
        for r in rows:
            v = r.get(col)
            if isinstance(v, (int, float)):
                nums.append(float(v))
            elif v is not None:
                try:
                    nums.append(float(str(v)))
                except (ValueError, TypeError):
                    pass
        if nums:
            summary[col] = {
                "count": len(nums),
                "sum": round(sum(nums), 2),
                "min": round(min(nums), 2),
                "max": round(max(nums), 2),
                "avg": round(sum(nums) / len(nums), 2),
            }
    return summary


def build_data_context(
    user_q: str,
    columns: list[str],
    rows: list[JsonMap],
    data_rows: list[JsonMap],
    data_note: str,
    summary: Mapping[str, dict[str, float]],
) -> str:
    """Build the shared data context string for LLM prompts."""
    parts = [
        f"User Question: {user_q}",
        f"Columns: {json_dumps(columns, default=str)}",
        f"Total rows: {len(rows)}",
    ]
    if summary:
        parts.append(f"Column stats (for KPI cards): {json_dumps(summary, ensure_ascii=False)}")
    parts.append(
        f"Data ({data_note}): {json_dumps(data_rows, ensure_ascii=False, default=str)[:4000]}"
    )
    return "\n".join(parts)


__all__ = [
    # Schemas
    "SqlResultDict",
    "TokenUsageDict",
    "ValidationDict",
    # Helpers
    "acc_tokens",
    "build_data_context",
    "compute_column_summary",
    "extract_json",
    "invoke_model",
    "is_debug",
    "needs_chart",
    "needs_table",
    "validate_sql_syntax",
]
