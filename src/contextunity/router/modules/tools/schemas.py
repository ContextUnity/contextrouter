"""Centralized type definitions and models for contextunity.router.tools.

This module stores all Pydantic models and TypedDicts used as schemas or
return types for the various LLM tools. Keeping them here avoids circular
imports and enforces a standard contract pattern (e.g. ``BaseToolResult``).
"""

from __future__ import annotations

from contextunity.core.types import JsonDict, JsonValue
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
    args: NotRequired[JsonDict]


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
    """A detailed step in the execution graph (rendered in contextunity.view)."""

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
    episode_id: NotRequired[str]
    episodes: NotRequired[list[JsonDict]]
    count: NotRequired[int]
    user_id: NotRequired[str]


class MemoryRecord(TypedDict, total=False):
    """A single memory record returned from semantic/episodic recall."""

    id: str
    content: str
    metadata: JsonDict
    score: float
    created_at: str


class MemoryRecallResult(BaseToolResult):
    """Result of fetching semantic/episodic memory."""

    memories: NotRequired[list[MemoryRecord]]
    content: NotRequired[str]
    facts: NotRequired[dict[str, str]]
    count: NotRequired[int]
    key: NotRequired[str]
    value: NotRequired[str]
    user_id: NotRequired[str]


# ============================================================================
# GCS Storage (gcs_tools)
# ============================================================================


class GCSBlobEntry(TypedDict):
    """Serialised blob metadata for GCS list operations."""

    name: str
    size: int | None
    content_type: str | None
    updated: str | None


class GCSResult(BaseToolResult):
    """Result of GCS storage operations (upload/download/list)."""

    status: NotRequired[str]
    gcs_uri: NotRequired[str]
    path: NotRequired[str]
    bucket: NotRequired[str]
    size: NotRequired[int]
    content: NotRequired[str]
    content_type: NotRequired[str]
    prefix: NotRequired[str]
    count: NotRequired[int]
    blobs: NotRequired[list[GCSBlobEntry]]
    found: NotRequired[bool]


# ============================================================================
# SQL (sql tools)
# ============================================================================


class SQLResult(BaseToolResult):
    """Result of SQL validation and execution."""

    valid: NotRequired[bool]
    sql: NotRequired[str]
    columns: NotRequired[list[str]]
    rows: NotRequired[list[list[JsonValue]]]
    row_count: NotRequired[int]
    elapsed_ms: NotRequired[float]
    sql_executed: NotRequired[str]


# ============================================================================
# Security (shield_scan, check_policy, check_compliance)
# ============================================================================


class SecurityResult(BaseToolResult):
    """Result of security operations (shield scan, policy check, compliance)."""

    # shield_scan fields
    allowed: NotRequired[bool]
    blocked: NotRequired[bool]
    threats: NotRequired[list[dict[str, str]]]
    risk_score: NotRequired[float]
    severity: NotRequired[str]
    latency_ms: NotRequired[float]

    # check_policy fields
    reason: NotRequired[str]
    matched_policy: NotRequired[str]
    evaluation_ms: NotRequired[float]

    # check_compliance fields
    compliant: NotRequired[bool]
    score: NotRequired[int | float]
    findings: NotRequired[list[dict[str, str]]]
    summary: NotRequired[str]


# ============================================================================
# Privacy (anonymize, check_pii)
# ============================================================================


class PrivacyResult(BaseToolResult):
    """Result of privacy operations (anonymize, check_pii)."""

    anonymized_text: NotRequired[str]
    entities_masked: NotRequired[int]
    entity_types: NotRequired[list[str]]
    session_id: NotRequired[str]
    persona_injected: NotRequired[bool]

    # check_pii fields
    contains_pii: NotRequired[bool]
    entities_found: NotRequired[int]


# ============================================================================
# Redis Memory (store, retrieve, cache, session, clear)
# ============================================================================


class RedisMemoryResult(BaseToolResult):
    """Result of Redis memory operations."""

    key: NotRequired[str]
    value: NotRequired[str]
    memory_key: NotRequired[str]
    cache_key: NotRequired[str]
    ttl_seconds: NotRequired[int]
    found: NotRequired[bool]
    timestamp: NotRequired[str]
    query: NotRequired[str]
    result: NotRequired[str]
    session_id: NotRequired[str]
    data: NotRequired[JsonDict]
    cleared: NotRequired[str]


# ============================================================================
# Sub-agent
# ============================================================================


class SubAgentResult(BaseToolResult):
    """Result of sub-agent spawning."""

    subagent_id: NotRequired[str | None]
    status: NotRequired[str]
    message: NotRequired[str]
    agent_type: NotRequired[str]
    strategy: NotRequired[str]


# ============================================================================
# Cognee
# ============================================================================


class CogneeResult(BaseToolResult):
    """Result of cognee graph extraction."""

    entities: NotRequired[list[JsonDict]]
    relations: NotRequired[list[JsonDict]]
    method: NotRequired[str]
