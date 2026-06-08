"""Test that ``on_chain_end`` handles non-dict outputs without crashing.

LangChain may invoke ``on_chain_end`` with a plain string (or any non-dict
payload) when a chain returns a primitive. The token-delta helper used to
call ``outputs.get(...)`` unconditionally, which raised
``AttributeError("'str' object has no attribute 'get'")``.
"""

from __future__ import annotations

import uuid

import pytest

from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.modules.observability.langchain_handlers import (
    LangchainCallbackMixin,
)


def _setup_span(tracer: BrainAutoTracer, run_id: uuid.UUID, parent_id: uuid.UUID) -> None:
    """Set up a chain span so ``on_chain_end`` does not bail early."""
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


@pytest.mark.asyncio
async def test_string_output_does_not_crash_on_chain_end() -> None:
    """Plain string output must be ignored gracefully — no AttributeError."""
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    _setup_span(tracer, run_id, parent_id)

    # Simulate a chain that returned a plain string
    await LangchainCallbackMixin.on_chain_end(
        tracer,
        "just a plain string result",  # not a dict
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    # result_json should hold the string serialization
    assert (
        "string" in span["result_json"]
        or "result" in span["result_json"]
        or len(span["result_json"]) > 0
    )
    # No token usage was recorded (no _token_usage in str)
    assert "tokens_in" not in span


@pytest.mark.asyncio
async def test_none_output_does_not_crash_on_chain_end() -> None:
    """A chain returning ``None`` must not crash either."""
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    _setup_span(tracer, run_id, parent_id)

    await LangchainCallbackMixin.on_chain_end(
        tracer,
        None,
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    # result_json is empty when outputs is None
    assert span["result_json"] == ""


@pytest.mark.asyncio
async def test_int_output_does_not_crash_on_chain_end() -> None:
    """Numeric primitive outputs (e.g. token counts) also handled."""
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    _setup_span(tracer, run_id, parent_id)

    await LangchainCallbackMixin.on_chain_end(
        tracer,
        42,
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    assert "42" in span["result_json"]


@pytest.mark.asyncio
async def test_dict_output_with_token_usage_still_works() -> None:
    """Smoke test: dict output with ``_token_usage`` still records delta."""
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    parent_id = uuid.uuid4()
    _setup_span(tracer, run_id, parent_id)

    await LangchainCallbackMixin.on_chain_end(
        tracer,
        {
            "_token_usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_cost": 0.001,
            }
        },
        run_id=run_id,
        parent_run_id=parent_id,
    )

    span = tracer.spans[str(run_id)]
    assert span["tokens_in"] == 10
    assert span["tokens_out"] == 5
    assert span["cost_usd"] == 0.001
