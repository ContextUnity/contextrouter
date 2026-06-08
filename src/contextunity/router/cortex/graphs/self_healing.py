"""Self-healing graph entrypoint (stub until YAML compiler template lands)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from contextunity.core.types import ContextUnitPayload


@runtime_checkable
class SelfHealingGraph(Protocol):
    """Minimal invoke surface used by View notification handler."""

    async def ainvoke(self, state: ContextUnitPayload) -> ContextUnitPayload: ...


class _StubSelfHealingGraph:
    """No-op graph — dispatcher agent notes full reimplementation is pending."""

    async def ainvoke(self, state: ContextUnitPayload) -> ContextUnitPayload:
        return {
            "healing_report": {
                "status": "skipped",
                "reason": "self_healing graph not yet reimplemented",
                "requested_keys": list(state.keys()),
            }
        }


def build_self_healing_graph() -> SelfHealingGraph:
    """Return the self-healing runnable graph."""
    return _StubSelfHealingGraph()
