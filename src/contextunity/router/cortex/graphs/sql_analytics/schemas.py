"""Schemas for SQL Analytics graph state."""

from typing import Any, TypedDict


class SqlResultDict(TypedDict, total=False):
    """Result of executing SQL query."""

    rows: list[dict[str, Any]]
    columns: list[str]
    duration_ms: float
    error: str


class ValidationDict(TypedDict, total=False):
    """Result of SQL data validation."""

    valid: bool
    reason: str
    hints: list[str]


class VisualizerComponent(TypedDict, total=False):
    """Single UI component returned by visualizer."""

    component: str
    title: str
    type: str
    data: Any
    config: Any


class TokenUsageDict(TypedDict, total=False):
    """Accumulated token usage tracking."""

    input_tokens: int
    output_tokens: int
    total_cost: float


class SqlAnalyticsStateUpdate(TypedDict, total=False):
    """Partial update for SqlAnalyticsState."""

    messages: list[Any]
    metadata: dict[str, Any]
    sql: str
    purpose: str
    format: str
    sql_result: SqlResultDict
    validation: ValidationDict
    components: list[VisualizerComponent]
    error: str
    retry_count: int
    _start_ts: float
    _token_usage: TokenUsageDict
