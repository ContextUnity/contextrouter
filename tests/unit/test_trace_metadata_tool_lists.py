"""End-to-end test: ``allowed_tools`` and ``denied_tools`` flow from the
execution input through the wire metadata that lands in Brain.

M11 (plan) flagged that the View dashboard's tool-restriction badges
showed empty for non-dispatcher executions. The propagation path is:

1. ``prepare_execution`` in ``helpers.py`` lifts ``allowed_tools`` /
   ``denied_tools`` from ``execution_input`` into ``metadata``.
2. ``_clean_for_trace`` preserves them (it only redacts
   ``project_config`` and Langfuse keys).
3. ``_trace_metadata`` in ``brain_trace_tools`` passes everything
   through to the Brain wire payload.
4. View dashboard ``traces/detail.py`` reads ``meta.get("allowed_tools")``
   to render the badges.

This test pins the contract end-to-end.
"""

from __future__ import annotations

from contextunity.router.modules.tools.brain_trace_tools import _trace_metadata
from contextunity.router.service.mixins.execution.helpers import _clean_for_trace
from contextunity.router.service.mixins.execution.metadata_helpers import (
    execution_metadata_from_payload,
)


def test_allowed_denied_tools_flow_to_wire_metadata() -> None:
    """Full chain: payload → metadata → clean → wire keeps tool lists."""
    meta = execution_metadata_from_payload(
        {
            "allowed_tools": ["search", "calc"],
            "denied_tools": ["delete_user"],
        }
    )
    # Step 1: payload extraction keeps the lists
    assert meta.get("allowed_tools") == ["search", "calc"]
    assert meta.get("denied_tools") == ["delete_user"]

    # Step 2: _clean_for_trace preserves them
    cleaned = _clean_for_trace(meta)
    assert cleaned.get("allowed_tools") == ["search", "calc"]
    assert cleaned.get("denied_tools") == ["delete_user"]

    # Step 3: _trace_metadata emits them on the wire
    wire = _trace_metadata(
        metadata=cleaned,
        model_key="openai/gpt-4o",
        platform="grpc",
        iterations=1,
        message_count=0,
        steps=None,
    )
    assert wire.get("allowed_tools") == ["search", "calc"]
    assert wire.get("denied_tools") == ["delete_user"]


def test_empty_tool_lists_are_preserved_as_empty_lists() -> None:
    """An empty ``allowed_tools=[]`` (meaning 'no tools allowed') must
    not be conflated with a missing field — both are valid signals."""
    meta = execution_metadata_from_payload({"allowed_tools": [], "denied_tools": []})
    cleaned = _clean_for_trace(meta)
    wire = _trace_metadata(
        metadata=cleaned,
        model_key="m",
        platform="grpc",
        iterations=1,
        message_count=0,
        steps=None,
    )
    assert wire.get("allowed_tools") == []
    assert wire.get("denied_tools") == []


def test_wildcard_allowed_tools_preserved() -> None:
    """``["*"]`` shorthand for 'all tools allowed' must survive the chain."""
    meta = execution_metadata_from_payload({"allowed_tools": ["*"]})
    wire = _trace_metadata(
        metadata=_clean_for_trace(meta),
        model_key="m",
        platform="grpc",
        iterations=1,
        message_count=0,
        steps=None,
    )
    assert wire.get("allowed_tools") == ["*"]


def test_metadata_drops_non_string_tool_entries() -> None:
    """Non-string entries must be filtered out (L2 JSON-cleanliness)."""
    meta = execution_metadata_from_payload(
        {
            "allowed_tools": ["valid", 123, None, "also_valid"],
        }
    )
    wire = _trace_metadata(
        metadata=_clean_for_trace(meta),
        model_key="m",
        platform="grpc",
        iterations=1,
        message_count=0,
        steps=None,
    )
    # Only the two valid strings survive
    assert wire.get("allowed_tools") == ["valid", "also_valid"]


def test_metadata_drops_request_controlled_langfuse_credentials() -> None:
    meta = execution_metadata_from_payload(
        {
            "langfuse_secret_key": "attacker-secret",
            "langfuse_public_key": "attacker-public",
            "langfuse_host": "http://attacker.invalid",
        }
    )

    assert "langfuse_secret_key" not in meta
    assert "langfuse_public_key" not in meta
    assert "langfuse_host" not in meta
