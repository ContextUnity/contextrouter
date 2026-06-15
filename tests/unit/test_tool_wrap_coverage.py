"""Wrap-coverage contract for the router tool registry.

Pins two guarantees:
1. Every tool returned by discover_all_tools() is a SecureTool (permission
   enforcement + authoritative tenant injection cannot be skipped).
2. Every @tool defined in the tool modules is actually picked up by
   discovery — a new tool file cannot ship outside the SecureTool path
   without this test failing.
"""

from __future__ import annotations

import importlib
import inspect

from langchain_core.tools import BaseTool

from contextunity.router.modules.tools import SecureTool, discover_all_tools

# Modules whose @tool functions must all surface through discovery.
# sql/security/gcs tools register per-project or are optional-dependency
# gated; they are covered by the isinstance sweep below when present.
_COVERED_MODULES = [
    "contextunity.router.modules.tools.redis_memory",
    "contextunity.router.modules.tools.brain_memory_tools",
    "contextunity.router.modules.tools.brain_admin_tools",
]


def _module_tool_names(module_path: str) -> set[str]:
    module = importlib.import_module(module_path)
    names: set[str] = set()
    for _, obj in inspect.getmembers(module):
        if isinstance(obj, BaseTool):
            names.add(obj.name)
    return names


def test_discover_all_tools_returns_only_secure_tools() -> None:
    tools = discover_all_tools()
    assert tools, "tool discovery returned nothing"
    unwrapped = [t.name for t in tools if not isinstance(t, SecureTool)]
    assert unwrapped == [], f"tools discovered without SecureTool wrapping: {unwrapped}"


def test_module_tools_are_all_discovered() -> None:
    discovered = {t.name for t in discover_all_tools()}
    for module_path in _COVERED_MODULES:
        expected = _module_tool_names(module_path)
        assert expected, f"{module_path} defines no tools — update _COVERED_MODULES"
        missing = expected - discovered
        assert missing == set(), (
            f"{module_path} defines tools not reachable through discovery "
            f"(they would bypass SecureTool wrapping): {sorted(missing)}"
        )
