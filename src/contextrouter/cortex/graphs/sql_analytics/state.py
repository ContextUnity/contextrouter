"""State definition for SQL analytics graph."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class SqlAnalyticsState(TypedDict):
    """State for SQL analytics graph.

    Flow: planner → execute_sql → verifier → visualizer → reflect → END

    Each LLM node handles PII atomically (anonymize→LLM→deanonymize).
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: dict[str, Any]

    # Pipeline data
    sql: str
    purpose: str
    format: str
    sql_result: dict[str, Any]
    validation: dict[str, Any]
    components: list[dict[str, Any]]

    # Control
    error: str
    retry_count: int
    _start_ts: float
    _token_usage: dict[str, Any]  # accumulated {input_tokens, output_tokens, total_cost}

    # Trace — each node appends its step record via operator.add reducer
    _steps: Annotated[list[dict[str, Any]], operator.add]


__all__ = ["SqlAnalyticsState"]
