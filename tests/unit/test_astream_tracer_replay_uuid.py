"""Tests for (UUID guard) in ``astream_tracer_replay``.

The original ``_as_uuid`` did ``UUID(str(value))`` which raises ``ValueError``
on empty or malformed input and crashes the StreamAgent replay loop in
production. The fix returns ``None`` for non-coercible values and the
caller skips the event.
"""

from __future__ import annotations

import uuid
from uuid import UUID

import pytest

from contextunity.router.modules.observability.astream_tracer_replay import (
    _as_uuid,
    replay_astream_event_to_tracer,
)
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer


def test_as_uuid_returns_none_for_empty_string() -> None:
    """Empty string must not raise — fail-closed contract."""
    assert _as_uuid("") is None


def test_as_uuid_returns_none_for_none() -> None:
    assert _as_uuid(None) is None


def test_as_uuid_returns_none_for_malformed_string() -> None:
    assert _as_uuid("not-a-uuid") is None
    assert _as_uuid("12345") is None


def test_as_uuid_returns_uuid_for_valid_string() -> None:
    valid = uuid.uuid4()
    result = _as_uuid(str(valid))
    assert result == valid
    assert isinstance(result, UUID)


def test_as_uuid_returns_uuid_for_uuid_instance() -> None:
    valid = uuid.uuid4()
    assert _as_uuid(valid) == valid


def test_as_uuid_returns_none_for_non_string_object() -> None:
    assert _as_uuid(12345) is None
    assert _as_uuid([1, 2, 3]) is None
    assert _as_uuid({"key": "value"}) is None


@pytest.mark.asyncio
async def test_replay_skips_event_with_empty_run_id() -> None:
    """Replay loop must not crash on missing run_id; it should skip the event."""
    tracer = BrainAutoTracer()
    event = {
        "event": "on_chain_start",
        "name": "LangGraph",
        "run_id": "",
        "parent_ids": [],
        "tags": [],
        "metadata": {},
        "data": {"input": {}},
    }
    await replay_astream_event_to_tracer(tracer, event)
    assert len(tracer.spans) == 0


@pytest.mark.asyncio
async def test_replay_skips_event_with_no_run_id_key() -> None:
    """Replay loop must not crash when run_id key is absent."""
    tracer = BrainAutoTracer()
    event = {
        "event": "on_chain_start",
        "name": "LangGraph",
        "parent_ids": [],
        "tags": [],
        "metadata": {},
        "data": {"input": {}},
    }
    await replay_astream_event_to_tracer(tracer, event)
    assert len(tracer.spans) == 0


@pytest.mark.asyncio
async def test_replay_processes_event_with_valid_run_id() -> None:
    """Replay should still work correctly for well-formed events."""
    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    event = {
        "event": "on_chain_start",
        "name": "LangGraph",
        "run_id": str(run_id),
        "parent_ids": [],
        "tags": [],
        "metadata": {},
        "data": {"input": {}},
    }
    await replay_astream_event_to_tracer(tracer, event)
    assert str(run_id) in tracer.spans
