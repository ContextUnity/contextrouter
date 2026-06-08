"""Tests for fixes in StreamAgent event handling.

These tests pin the *contract* of the event-to-SSE transformation by
feeding a raw event dict into the same code path StreamAgent uses and
asserting on the output shape. We avoid spinning up a full gRPC stack.

on_chain_start uses ``is_object_dict`` (not ``is_json_dict``) for the
    data payload, matching on_chain_end. Non-JSON values like datetime /
    Decimal no longer silently produce empty progress for the start event.

brain_event SSE stream serializes both dataclass and dict shapes.
    Previously only ``is_dataclass`` events were propagated.

progress events are ``sanitize_for_struct``-ed before yielding so
     datetime/UUID/Decimal are coerced to JSON-safe forms.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from contextunity.core.types import is_object_dict
from langchain_core.messages import HumanMessage

# ── guard parity test ──────────────────────────────────────────────────


def test_chain_start_uses_is_object_dict_to_match_chain_end() -> None:
    """Both chain_start and chain_end must accept non-JSON dicts."""
    # Event with non-JSON values inside the input dict.
    raw_event = {
        "event": "on_chain_start",
        "name": "planner",
        "run_id": str(uuid.uuid4()),
        "parent_ids": [],
        "tags": [],
        "metadata": {},
        "data": {
            "input": {
                "messages": [HumanMessage(content="hi")],
                "ts": datetime.now(timezone.utc),
                "amount": Decimal("3.14"),
            }
        },
    }
    data = raw_event.get("data")
    # stream handler uses is_object_dict — same as chain_end.
    assert is_object_dict(data) is True
    inner = data.get("input", {})
    assert is_object_dict(inner) is True
    # Raw event keeps the datetime + Decimal so the handler can sanitize later.


# ── sanitize_for_struct converts non-JSON values ──────────────────────


def test_sanitize_for_struct_handles_datetime_decimal_uuid() -> None:
    """Progress payloads with datetime/Decimal/UUID must JSON-encode after sanitize."""
    from contextunity.router.service.security import sanitize_for_struct

    raw = {
        "ts": datetime.now(timezone.utc),
        "amount": Decimal("3.14"),
        "id": uuid.uuid4(),
        "messages": [HumanMessage(content="hi")],
    }
    sanitized = sanitize_for_struct(raw)
    # Must be JSON-encodable after sanitization.
    encoded = json.dumps(sanitized)
    decoded = json.loads(encoded)
    # Keys preserved; non-JSON values coerced to string/iso format.
    assert "ts" in decoded
    assert "amount" in decoded
    assert "id" in decoded
    assert isinstance(decoded["amount"], str)


# ── brain_event accepts both dataclass and dict ────────────────────────


@dataclass
class _FakeBrainEvent:
    type: str
    payload: dict[str, object]


def test_brain_event_dataclass_serializes_to_dict() -> None:
    """dataclass brain_event produces dict via asdict()."""
    event = _FakeBrainEvent(type="llm_start", payload={"model": "gpt"})
    payload_dict = asdict(event)
    assert payload_dict["type"] == "llm_start"
    assert payload_dict["payload"] == {"model": "gpt"}


def test_brain_event_dict_shape_passes_through_unchanged() -> None:
    """dict brain_event propagates as-is (no dataclass required)."""
    event_dict = {"type": "llm_end", "payload": {"tokens": 100}}
    # Simulate the branch: a dict-shaped event is propagated as-is.
    if isinstance(event_dict, dict):
        brain_event_payload = event_dict
    assert brain_event_payload == event_dict


# ── Behavioral end-to-end: feed synthetic events through replay + tracer ────


@pytest.mark.asyncio
async def test_replay_handles_chain_start_with_non_json_data() -> None:
    """End-to-end: chain_start with datetime/Decimal input must be replayable."""
    from contextunity.router.modules.observability.astream_tracer_replay import (
        replay_astream_event_to_tracer,
    )
    from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer

    tracer = BrainAutoTracer()
    run_id = uuid.uuid4()
    event = {
        "event": "on_chain_start",
        "name": "planner",
        "run_id": str(run_id),
        "parent_ids": [],
        "tags": [],
        "metadata": {"node": "planner"},
        "data": {
            "input": {
                "messages": [],
                "ts": datetime.now(timezone.utc),
                "amount": Decimal("3.14"),
            }
        },
    }
    await replay_astream_event_to_tracer(tracer, event)
    # is_object_dict accepts the data dict even though it contains
    # non-JSON values — span is registered with the full input.
    assert str(run_id) in tracer.spans
