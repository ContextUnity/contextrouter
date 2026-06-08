"""Tests for stream-mode graph state merge and astream trace replay."""

from __future__ import annotations

import uuid

import pytest

from contextunity.router.cortex.events import BrainEvent
from contextunity.router.cortex.types import merge_graph_state_update
from contextunity.router.modules.observability.astream_tracer_replay import (
    replay_astream_event_to_tracer,
)
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.service.mixins.execution.helpers import (
    extract_state_update_from_chain_output,
)


def test_extract_state_update_accepts_non_json_row_values():
    from datetime import datetime

    update = extract_state_update_from_chain_output(
        {
            "intermediate_results": {
                "tool_execution": {
                    "rows": [{"created_at": datetime(2024, 1, 2, 12, 0, 0)}],
                }
            },
            "final_output": {"rows": [{"created_at": datetime(2024, 1, 2, 12, 0, 0)}]},
        }
    )
    assert "tool_execution" in update["intermediate_results"]
    assert update["final_output"]["rows"][0]["created_at"] == datetime(2024, 1, 2, 12, 0, 0)


def test_trace_metadata_preserves_steps_when_project_config_present():
    from uuid import uuid4

    from contextunity.router.modules.tools import brain_trace_tools

    steps = [
        {
            "id": uuid4(),
            "is_group": True,
            "node": "planner",
            "type": "chain",
            "children": [
                {
                    "id": uuid4(),
                    "tool": "openai/gpt-5-mini",
                    "type": "assistant",
                    "status": "ok",
                }
            ],
        }
    ]
    wire = brain_trace_tools._trace_metadata(
        metadata={
            "project_config": {"graph": object(), "nodes": [{"prompt_ref": "x"}]},
            "platform": "grpc",
        },
        model_key="openai/gpt-5-mini",
        platform="grpc",
        iterations=1,
        message_count=2,
        steps=steps,
    )

    assert wire.get("platform") == "grpc"
    assert "project_config" not in wire
    assert isinstance(wire.get("steps"), list)
    assert len(wire["steps"]) == 1
    assert wire["steps"][0]["node"] == "planner"
    assert isinstance(wire["steps"][0]["id"], str)
    assert wire["steps"][0]["children"][0]["tool"] == "openai/gpt-5-mini"


def test_merge_graph_state_update_merges_messages_when_left_empty():
    from langchain_core.messages import AIMessage

    accumulated: dict[str, object] = {}
    update: dict[str, object] = {"messages": [AIMessage(content="ok")]}
    merged = merge_graph_state_update(accumulated, update)
    messages = merged.get("messages")
    assert isinstance(messages, list)
    assert len(messages) == 1


def test_merge_graph_state_update_shallow_merges_intermediate_results():
    accumulated = {
        "intermediate_results": {"planner": {"sql": "SELECT 1"}},
        "final_output": {"purpose": "test"},
    }
    update = {
        "intermediate_results": {"tool_execution": {"rows": [{"a": 1}]}},
        "final_output": {"rows": [{"a": 1}]},
    }

    merged = merge_graph_state_update(accumulated, update)

    assert merged["intermediate_results"] == {
        "planner": {"sql": "SELECT 1"},
        "tool_execution": {"rows": [{"a": 1}]},
    }
    assert merged["final_output"] == {"rows": [{"a": 1}]}


@pytest.mark.asyncio
async def test_replay_astream_event_populates_node_groups_and_tool_calls():
    tracer = BrainAutoTracer()
    graph_run_id = uuid.uuid4()
    planner_run_id = uuid.uuid4()

    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_chain_start",
            "name": "LangGraph",
            "run_id": str(graph_run_id),
            "parent_ids": [],
            "data": {"input": {}},
        },
    )
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_chain_start",
            "name": "planner",
            "run_id": str(planner_run_id),
            "parent_ids": [str(graph_run_id)],
            "data": {"input": {}},
        },
    )
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_chain_end",
            "name": "planner",
            "run_id": str(planner_run_id),
            "parent_ids": [str(graph_run_id)],
            "data": {"output": {"final_output": {"sql": "SELECT 1"}}},
        },
    )
    await replay_astream_event_to_tracer(
        tracer,
        {
            "event": "on_custom_event",
            "name": "brain_event",
            "run_id": str(uuid.uuid4()),
            "parent_ids": [str(planner_run_id)],
            "data": {
                "event": BrainEvent(
                    type="tool_result",
                    node="tool_execution",
                    data={
                        "status": "ok",
                        "duration_ms": 42,
                        "tool_binding": "federated:medical_sql",
                        "result": {"rows": [{"id": 1}]},
                    },
                )
            },
        },
    )

    steps = tracer.get_nested_steps()
    assert steps
    assert steps[0]["node"] == "planner"
    tool_calls = tracer.get_tool_calls_summary()
    assert tool_calls == [{"tool": "federated:medical_sql", "status": "ok"}]
