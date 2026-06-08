"""Dispatcher routing respects max_iterations."""

from __future__ import annotations

from contextunity.router.cortex.dispatcher_agent.routing import should_execute_tools


def _state(*, iteration: int, max_iterations: int = 2) -> dict[str, object]:
    return {
        "messages": [],
        "iteration": iteration,
        "max_iterations": max_iterations,
    }


def test_should_execute_tools_ends_when_iteration_cap_reached() -> None:
    assert should_execute_tools(_state(iteration=2, max_iterations=2)) == "end"
