"""Centralized type definitions and models for ContextRouter tools.

This module stores all Pydantic models and TypedDicts used as schemas or
return types for the various LLM tools. Keeping them here avoids circular
imports and enforces a standard contract pattern (e.g. ``BaseToolResult``).
"""

from __future__ import annotations

from typing import Any

from typing_extensions import NotRequired, TypedDict


class BaseToolResult(TypedDict):
    """The fundamental contract for all tool returns."""

    success: bool
    error: NotRequired[str]


# ============================================================================
# Observability & Tracing (brain_trace_tools)
# ============================================================================


class ToolCallSummary(TypedDict):
    """Summary of a single tool invocation."""

    tool: str
    status: NotRequired[str]
    args: NotRequired[dict[str, Any]]


class TraceTokens(TypedDict):
    """Token usage for a single evaluation step."""

    input: int
    output: int


class TotalTokenUsage(TypedDict, total=False):
    """Total token usage for a graph run."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float


class TraceStep(TypedDict):
    """A detailed step in the execution graph (rendered in ContextView)."""

    step: int
    iteration: int
    type: str  # 'system', 'user', 'assistant', 'tool_call', 'tool_result'
    content: NotRequired[str]
    tool: NotRequired[str]
    tool_call_id: NotRequired[str]
    args: NotRequired[dict[str, str]]
    tokens: NotRequired[TraceTokens]
    status: NotRequired[str]
    result: NotRequired[str]


class TraceResult(BaseToolResult):
    """Result of logging a full execution trace."""

    trace_id: NotRequired[str]
    tenant_id: NotRequired[str]
    graph_name: NotRequired[str]


class EpisodeResult(BaseToolResult):
    """Result of recording an episodic memory."""

    episode_id: NotRequired[str]


# ============================================================================
# Memory (brain_memory_tools & redis_memory)
# ============================================================================


class MemoryStoreResult(BaseToolResult):
    """Result of storing short/long term context or knowledge."""

    ids: NotRequired[list[str]]
    keys: NotRequired[list[str]]
    knowledge_id: NotRequired[str]


class MemoryRecord(TypedDict, total=False):
    """A single memory record returned from semantic/episodic recall."""

    id: str
    content: str
    metadata: dict[str, Any]
    score: float
    created_at: str


class MemoryRecallResult(BaseToolResult):
    """Result of fetching semantic/episodic memory."""

    memories: NotRequired[list[MemoryRecord]]
    content: NotRequired[str]


# ============================================================================
# Tool Data (sql, gcs, privacy, security)
# ============================================================================


class DataToolResult(BaseToolResult):
    """Result of fetching or generating data payloads."""

    data: NotRequired[Any]
    details: NotRequired[dict[str, Any]]
    results: NotRequired[list[Any]]
    url: NotRequired[str]
