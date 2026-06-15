"""Tests for the per-node token-usage delta logic.

``BrainAutoTracer.on_chain_end`` derives a node's token contribution by
comparing the cumulative ``_token_usage`` carried in the chain output
against ``last_token_usage``. This pins that contract.
"""

from __future__ import annotations

import uuid

import pytest

from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.modules.observability.langchain_handlers import (
    LangchainCallbackMixin,
)


def _make_tracer() -> BrainAutoTracer:
    return BrainAutoTracer()


@pytest.mark.asyncio
async def test_first_chain_end_records_full_amount_as_delta() -> None:
    """First call: ``last_token_usage`` is empty, so delta equals current."""
    tracer = _make_tracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()

    # Set up parent span so chain_end doesn't bail on missing parent
    tracer.spans[str(parent_id)] = {"ignore": True, "children": tracer.root_spans}
    # Set up child span
    tracer.spans[str(run_id)] = {
        "is_group": True,
        "node": "planner",
        "type": "chain",
        "status": "ok",
        "start_time": 0.0,
        "args_json": "",
        "children": [],
        "cumulative_ms": 0,
        "cumulative_usd": 0.0,
        "cumulative_tokens": 0,
        "timing_ms": 0,
        "has_result": False,
    }

    outputs = {
        "_token_usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_cost": 0.005,
        }
    }
    await LangchainCallbackMixin.on_chain_end(
        tracer,
        outputs,
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    assert span["tokens_in"] == 100
    assert span["tokens_out"] == 50
    assert span["cost_usd"] == 0.005
    # Running total recorded
    assert tracer.last_token_usage == {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_cost": 0.005,
    }


@pytest.mark.asyncio
async def test_subsequent_chain_end_records_delta_only() -> None:
    """Second call: delta is the increment, not the cumulative total."""
    tracer = _make_tracer()
    # Pre-seed the running total
    tracer.last_token_usage = {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_cost": 0.005,
    }

    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    tracer.spans[str(parent_id)] = {"ignore": True, "children": tracer.root_spans}
    tracer.spans[str(run_id)] = {
        "is_group": True,
        "node": "executor",
        "type": "chain",
        "status": "ok",
        "start_time": 0.0,
        "args_json": "",
        "children": [],
        "cumulative_ms": 0,
        "cumulative_usd": 0.0,
        "cumulative_tokens": 0,
        "timing_ms": 0,
        "has_result": False,
    }

    outputs = {
        "_token_usage": {
            "input_tokens": 200,  # +100 from previous
            "output_tokens": 100,  # +50 from previous
            "total_cost": 0.015,  # +0.01 from previous
        }
    }
    await LangchainCallbackMixin.on_chain_end(
        tracer,
        outputs,
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    assert span["tokens_in"] == 100
    assert span["tokens_out"] == 50
    assert span["cost_usd"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_chain_end_handles_missing_token_usage() -> None:
    """No ``_token_usage`` in outputs: span stays clean, no crash."""
    tracer = _make_tracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    tracer.spans[str(parent_id)] = {"ignore": True, "children": tracer.root_spans}
    tracer.spans[str(run_id)] = {
        "is_group": True,
        "node": "x",
        "type": "chain",
        "status": "ok",
        "start_time": 0.0,
        "args_json": "",
        "children": [],
        "cumulative_ms": 0,
        "cumulative_usd": 0.0,
        "cumulative_tokens": 0,
        "timing_ms": 0,
        "has_result": False,
    }

    await LangchainCallbackMixin.on_chain_end(
        tracer,
        {"final_output": {"content": "ok"}},
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    # No token fields set when usage is absent
    assert "tokens_in" not in span
    assert "tokens_out" not in span
    # Running total unchanged
    assert tracer.last_token_usage == {}


@pytest.mark.asyncio
async def test_chain_end_reads_token_usage_from_update_envelope() -> None:
    """LangGraph may wrap state update in ``outputs["update"]`` — handle that too."""
    tracer = _make_tracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    tracer.spans[str(parent_id)] = {"ignore": True, "children": tracer.root_spans}
    tracer.spans[str(run_id)] = {
        "is_group": True,
        "node": "x",
        "type": "chain",
        "status": "ok",
        "start_time": 0.0,
        "args_json": "",
        "children": [],
        "cumulative_ms": 0,
        "cumulative_usd": 0.0,
        "cumulative_tokens": 0,
        "timing_ms": 0,
        "has_result": False,
    }

    outputs = {
        "update": {
            "_token_usage": {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_cost": 0.001,
            }
        }
    }
    await LangchainCallbackMixin.on_chain_end(
        tracer,
        outputs,
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    assert span["tokens_in"] == 7
    assert span["tokens_out"] == 3
    assert span["cost_usd"] == 0.001


@pytest.mark.asyncio
async def test_chain_end_propagates_token_delta_to_assistant_child() -> None:
    """Per-node delta on the chain group is copied to a zeroed assistant span."""
    tracer = _make_tracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    assistant_id = uuid.uuid4()

    tracer.spans[str(parent_id)] = {"ignore": True, "children": tracer.root_spans}
    tracer.spans[str(run_id)] = {
        "is_group": True,
        "node": "planner",
        "type": "chain",
        "status": "ok",
        "start_time": 0.0,
        "args_json": "",
        "children": [
            {
                "id": assistant_id,
                "is_group": False,
                "tool": "openai/gpt-5-mini",
                "type": "assistant",
                "status": "ok",
                "tokens_in": 0,
                "tokens_out": 0,
                "tokens": 0,
            },
        ],
        "cumulative_ms": 0,
        "cumulative_usd": 0.0,
        "cumulative_tokens": 0,
        "timing_ms": 0,
        "has_result": False,
    }

    await LangchainCallbackMixin.on_chain_end(
        tracer,
        {
            "_token_usage": {
                "input_tokens": 800,
                "output_tokens": 200,
                "total_cost": 0.002,
            }
        },
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    assistant = span["children"][0]
    assert span["tokens_in"] == 800
    assert span["tokens_out"] == 200
    assert assistant["tokens_in"] == 800
    assert assistant["tokens_out"] == 200
    assert assistant["tokens"] == 1000
