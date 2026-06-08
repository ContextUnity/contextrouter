"""Dispatcher agent/tools nodes share the same allowed_tools contract."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from contextunity.router.cortex.dispatcher_agent.tool_resolution import (
    dispatcher_tools_for_state,
)


def test_empty_allowed_tools_means_no_tools() -> None:
    assert dispatcher_tools_for_state({"allowed_tools": []}) == []


def test_wildcard_allowed_tools_means_all_discovered() -> None:
    tool_a = MagicMock()
    tool_a.name = "brain_search"

    with patch(
        "contextunity.router.cortex.dispatcher_agent.tool_resolution.dispatcher_available_tools_for_state",
        return_value=[tool_a],
    ):
        resolved = dispatcher_tools_for_state({"allowed_tools": ["*"]})

    assert len(resolved) == 1


def test_explicit_allow_list_filters_tools() -> None:
    tool_a = MagicMock()
    tool_a.name = "brain_search"
    tool_b = MagicMock()
    tool_b.name = "sql"

    with patch(
        "contextunity.router.cortex.dispatcher_agent.tool_resolution.dispatcher_available_tools_for_state",
        return_value=[tool_a, tool_b],
    ):
        resolved = dispatcher_tools_for_state({"allowed_tools": ["sql"]})

    assert [t.name for t in resolved] == ["sql"]
