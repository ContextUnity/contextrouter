"""Prompt-version propagation into BrainAutoTracer spans."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from contextunity.router.cortex.events import BrainEvent
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer


@pytest.mark.asyncio
async def test_custom_brain_event_records_prompt_version():
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    tracer.spans[run_id] = {
        "is_group": True,
        "node": "planner",
        "children": [],
        "ignore": False,
    }

    await tracer.on_custom_event(
        "brain_event",
        {
            "event": BrainEvent(
                type="llm_start",
                node="planner",
                data={
                    "model": "openai/gpt-5-mini",
                    "args": "System prompt",
                    "prompt_version": "deadbeef",
                },
            )
        },
        run_id=run_id,
    )

    child = tracer.spans[run_id]["children"][0]
    assert child["prompt_version"] == "deadbeef"


@pytest.mark.asyncio
async def test_langchain_chat_start_records_prompt_version_metadata():
    tracer = BrainAutoTracer()

    class _Message:
        type = "system"
        content = "System prompt"

    await tracer.on_chat_model_start(
        {"name": "inception/mercury-2"},
        [[_Message()]],
        run_id="run-1",
        metadata={"prompt_version": "cafebabe"},
    )

    assert tracer.spans["run-1"]["prompt_version"] == "cafebabe"


@pytest.mark.asyncio
async def test_invoke_model_passes_prompt_version_to_callbacks():
    from contextunity.router.cortex.compiler.platform_tools.helpers.sql import invoke_model
    from contextunity.router.modules.models.types import ModelResponse, ProviderInfo, UsageStats

    tracer = BrainAutoTracer()

    class _Model:
        name = "test/model"

        async def generate(self, request, **kwargs):
            return ModelResponse(
                text='{"ok": true}',
                usage=UsageStats(input_tokens=3, output_tokens=4, total_tokens=7),
                raw_provider=ProviderInfo(
                    provider="test", model_name="test/model", model_key="test/model"
                ),
            )

    await invoke_model(
        _Model(),
        [SimpleNamespace(type="system", content="System prompt")],
        config={"callbacks": [tracer]},
        prompt_version="0123abcd",
    )

    assert tracer.root_spans[0]["prompt_version"] == "0123abcd"
