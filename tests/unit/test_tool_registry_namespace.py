"""Tests for per-project tool namespace (plan).

two projects can register distinct tools under the same name
without silent overwrites. ``register_tool(project_id=...)`` indexes
into ``_tool_registry_by_project`` keyed by ``(project_id, name)``.
``get_tool_for_project`` returns the project-scoped tool.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import BaseTool

from contextunity.router.modules.tools import (
    deregister_tool,
    discover_tools_for_project,
    get_tool,
    get_tool_for_project,
    list_project_tools,
    register_tool,
)


class _Echo(BaseTool):
    name: str = "echo_h7"
    description: str = "echo"

    def _run(self, **kwargs: object) -> str:
        return "echo"


class _EchoAlt(BaseTool):
    name: str = "echo_h7"  # SAME name, different implementation
    description: str = "alternative echo"

    def _run(self, **kwargs: object) -> str:
        return "echo-alt"


@pytest.fixture(autouse=True)
def _cleanup_registry() -> None:
    """Ensure each test starts and ends with a clean registry state for our tool name."""
    deregister_tool("echo_h7")
    deregister_tool("echo_h7", project_id="proj_a")
    deregister_tool("echo_h7", project_id="proj_b")
    yield
    deregister_tool("echo_h7")
    deregister_tool("echo_h7", project_id="proj_a")
    deregister_tool("echo_h7", project_id="proj_b")


def test_register_tool_with_project_id_indexes_per_project() -> None:
    inner = _Echo()
    register_tool(inner, project_id="proj_a")
    # get_tool_for_project returns the project-scoped instance.
    tool_a = get_tool_for_project("proj_a", "echo_h7")
    assert tool_a is not None
    assert tool_a.name == "echo_h7"
    assert get_tool("echo_h7") is None


def test_get_tool_for_project_falls_back_to_global() -> None:
    """A tool registered without project_id is still reachable via fallback."""
    inner = _Echo()
    register_tool(inner)
    # No project-scoped entry; get_tool_for_project falls back to global.
    tool = get_tool_for_project("any_project", "echo_h7")
    assert tool is not None
    assert tool.name == "echo_h7"


def test_two_projects_can_have_distinct_tools_under_same_name() -> None:
    """distinct implementations under the same tool name must coexist."""
    register_tool(_Echo(), project_id="proj_a")
    register_tool(_EchoAlt(), project_id="proj_b")

    tool_a = get_tool_for_project("proj_a", "echo_h7")
    tool_b = get_tool_for_project("proj_b", "echo_h7")
    assert tool_a is not None and tool_b is not None
    # They are distinct SecureTool instances wrapping different _run methods.
    assert tool_a is not tool_b


def test_project_discovery_exposes_only_requested_project_tool() -> None:
    register_tool(_Echo(), project_id="proj_a")
    register_tool(_EchoAlt(), project_id="proj_b")

    discovered_a = {tool.name: tool for tool in discover_tools_for_project("proj_a")}
    discovered_b = {tool.name: tool for tool in discover_tools_for_project("proj_b")}

    assert discovered_a["echo_h7"] is get_tool_for_project("proj_a", "echo_h7")
    assert discovered_b["echo_h7"] is get_tool_for_project("proj_b", "echo_h7")
    assert discovered_a["echo_h7"] is not discovered_b["echo_h7"]


def test_list_project_tools_returns_only_project_scoped() -> None:
    register_tool(_Echo(), project_id="proj_a")
    register_tool(_EchoAlt(), project_id="proj_a")
    register_tool(_Echo(), project_id="proj_b")

    proj_a_tools = list_project_tools("proj_a")
    assert set(proj_a_tools) == {"echo_h7"}
    proj_b_tools = list_project_tools("proj_b")
    assert set(proj_b_tools) == {"echo_h7"}


def test_deregister_tool_with_project_id_only_removes_project_slot() -> None:
    register_tool(_Echo(), project_id="proj_a")
    register_tool(_EchoAlt(), project_id="proj_b")

    assert deregister_tool("echo_h7", project_id="proj_a") is True
    # proj_a slot removed from per-project registry
    assert "echo_h7" not in list_project_tools("proj_a")
    # proj_b slot intact
    assert "echo_h7" in list_project_tools("proj_b")


def test_deregister_project_tool_returns_false_when_missing() -> None:
    assert deregister_tool("echo_h7", project_id="proj_a") is False
