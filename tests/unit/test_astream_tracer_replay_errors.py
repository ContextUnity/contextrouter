"""Replay forwards LangGraph error events into BrainAutoTracer."""

from __future__ import annotations

import uuid

import pytest

from contextunity.router.modules.observability.astream_tracer_replay import (
    replay_astream_event_to_tracer,
)
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer


@pytest.mark.asyncio
async def test_replay_on_chain_error_marks_span_error() -> None:
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_chain_start",
            "name": "planner",
            "run_id": str(run_id),
            "parent_ids": [],
            "data": {"input": {}},
        },
    )
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_chain_error",
            "name": "planner",
            "run_id": str(run_id),
            "parent_ids": [],
            "data": {"error": ValueError("node failed")},
        },
    )
    span = tracer.spans[str(run_id)]
    assert span["status"] == "error"


@pytest.mark.asyncio
async def test_replay_on_tool_error_marks_span_error() -> None:
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_tool_start",
            "name": "medical_sql",
            "run_id": str(run_id),
            "parent_ids": [],
            "data": {"serialized": {"name": "medical_sql"}, "input": "{}"},
        },
    )
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_tool_error",
            "name": "medical_sql",
            "run_id": str(run_id),
            "parent_ids": [],
            "data": {"error": RuntimeError("tool timeout")},
        },
    )
    span = tracer.spans[str(run_id)]
    assert span["status"] == "error"
