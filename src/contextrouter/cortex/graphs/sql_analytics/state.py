"""State definition for SQL analytics graph."""

from __future__ import annotations

from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from contextrouter.cortex.graphs.sql_analytics.schemas import (
    SqlResultDict,
    TokenUsageDict,
    ValidationDict,
    VisualizerComponent,
)


class SqlAnalyticsState(TypedDict):
    """State for SQL analytics graph.

    Flow: planner → execute_sql → verifier → visualizer → reflect → END

    Each LLM node handles PII atomically (anonymize→LLM→deanonymize).
    Tracing is handled by BrainAutoTracer via LangChain callbacks.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: dict[str, Any]

    # Pipeline data
    sql: str
    purpose: str
    format: str
    sql_result: SqlResultDict
    validation: ValidationDict
    components: list[VisualizerComponent]

    # Control
    error: str
    retry_count: int
    _start_ts: float
    _token_usage: TokenUsageDict  # accumulated {input_tokens, output_tokens, total_cost}


__all__ = ["SqlAnalyticsState"]
