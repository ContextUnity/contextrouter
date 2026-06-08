"""RED phase: Tests for platform tool registration wiring.

Verifies that register_all_platform_tools() registers every expected binding.
"""

from __future__ import annotations

from contextunity.router.cortex.compiler.platform_registry import (
    PlatformToolRegistry,
)


class TestRegisterAllPlatformToolsWiring:
    """Verify register_all_platform_tools() wires every expected binding."""

    _ALL_EXPECTED_BINDINGS = [
        # Content
        "router_classify",
        "router_generate_content",
        "router_review_content",
        "router_filter_content",
        "router_plan_content",
        "router_match_semantic",
        # RLM
        "router_rlm_process",
        # Brain
        "brain_search",
        "brain_memory_read",
        "brain_memory_write",
        "brain_blackboard_write",
        "brain_blackboard_read",
        "brain_kg_query",
        "brain_upsert",
        # Shield
        "shield_scan",
        # Worker
        "worker_start_workflow",
        "worker_get_status",
        "worker_execute_code",
        "worker_register_schedules",
        # Language
        "language_tool",
    ]

    def test_all_bindings_registered(self):
        """Every platform binding must be resolvable after register_all_platform_tools()."""
        from contextunity.router.cortex.compiler.platform_tools import (
            register_all_platform_tools,
        )

        registry = PlatformToolRegistry()
        register_all_platform_tools(registry)
        missing = [b for b in self._ALL_EXPECTED_BINDINGS if not registry.has(b)]
        assert not missing, f"Missing bindings: {missing}"
