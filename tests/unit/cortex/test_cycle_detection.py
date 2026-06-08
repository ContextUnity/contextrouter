"""Tests for cycle detection, atomic graph hot reload, provenance HMAC.

TDD: RED phase — tests before implementation.
"""

import pytest


class TestCycleDetection:
    """Back-edges require config.max_retries to prevent infinite loops."""

    def test_simple_cycle_without_max_retries_rejected(self):
        """A→B→A without max_retries must be rejected."""
        from contextunity.router.core.exceptions import RouterGraphBuilderError
        from contextunity.router.cortex.compiler.builder import (
            build_local_graph,
        )

        nodes = [
            {"name": "node_a", "type": "llm", "config": {}},
            {"name": "node_b", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "node_a"},
            {"from_node": "node_a", "to_node": "node_b"},
            {"from_node": "node_b", "to_node": "node_a"},  # back-edge → cycle
        ]
        config = {"services": {}}

        with pytest.raises(RouterGraphBuilderError, match="cycle"):
            build_local_graph(nodes, edges, config, router_defaults={})

    def test_cycle_with_max_retries_accepted(self):
        """A→B→A with max_retries is acceptable (controlled retry loop)."""
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [
            {"name": "node_a", "type": "llm", "config": {}},
            {"name": "node_b", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "node_a"},
            {"from_node": "node_a", "to_node": "node_b"},
            {"from_node": "node_b", "to_node": "node_a"},
        ]
        config = {"max_retries": 3, "services": {}}

        # Should not raise — max_retries provides the guard
        _validate_manifest_security(nodes, edges, config)

    def test_conditional_back_edge_detected(self):
        """Conditional edge creating a cycle also requires max_retries."""
        from contextunity.router.core.exceptions import RouterGraphBuilderError
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [
            {"name": "generator", "type": "llm", "config": {}},
            {"name": "reflector", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "generator"},
            {"from_node": "generator", "to_node": "reflector"},
            {
                "from_node": "reflector",
                "condition_key": "needs_retry",
                "condition_map": {
                    "true": "generator",  # back-edge
                    "false": "__end__",
                },
            },
        ]
        config = {"services": {}}

        with pytest.raises(RouterGraphBuilderError, match="cycle"):
            _validate_manifest_security(nodes, edges, config)

    def test_conditional_back_edge_with_max_retries_accepted(self):
        """Conditional back-edge with max_retries is acceptable."""
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [
            {"name": "generator", "type": "llm", "config": {}},
            {"name": "reflector", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "generator"},
            {"from_node": "generator", "to_node": "reflector"},
            {
                "from_node": "reflector",
                "condition_key": "needs_retry",
                "condition_map": {
                    "true": "generator",
                    "false": "__end__",
                },
            },
        ]
        config = {"max_retries": 2, "services": {}}

        # Should not raise
        _validate_manifest_security(nodes, edges, config)

    def test_cycle_with_float_max_retries_accepted(self):
        """Whole-number float max_retries from gRPC wire payloads is accepted."""
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [
            {"name": "node_a", "type": "llm", "config": {}},
            {"name": "node_b", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "node_a"},
            {"from_node": "node_a", "to_node": "node_b"},
            {"from_node": "node_b", "to_node": "node_a"},
        ]
        config = {"max_retries": 2.0, "services": {}}

        _validate_manifest_security(nodes, edges, config)

    def test_cycle_with_zero_max_retries_accepted(self):
        """max_retries=0 is an explicit guard (no retries), not a missing value."""
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [
            {"name": "node_a", "type": "llm", "config": {}},
            {"name": "node_b", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "node_a"},
            {"from_node": "node_a", "to_node": "node_b"},
            {"from_node": "node_b", "to_node": "node_a"},
        ]
        config = {"max_retries": 0, "services": {}}

        _validate_manifest_security(nodes, edges, config)

    def test_linear_graph_no_cycle(self):
        """Linear graph (no back-edges) should always pass."""
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [
            {"name": "node_a", "type": "llm", "config": {}},
            {"name": "node_b", "type": "llm", "config": {}},
            {"name": "node_c", "type": "llm", "config": {}},
        ]
        edges = [
            {"from_node": "__start__", "to_node": "node_a"},
            {"from_node": "node_a", "to_node": "node_b"},
            {"from_node": "node_b", "to_node": "node_c"},
            {"from_node": "node_c", "to_node": "__end__"},
        ]
        config = {"services": {}}

        # Should not raise
        _validate_manifest_security(nodes, edges, config)

    def test_self_loop_detected(self):
        """Self-loop (A→A) is a cycle and requires max_retries."""
        from contextunity.router.core.exceptions import RouterGraphBuilderError
        from contextunity.router.cortex.compiler.builder import (
            _validate_manifest_security,
        )

        nodes = [{"name": "looper", "type": "llm", "config": {}}]
        edges = [
            {"from_node": "__start__", "to_node": "looper"},
            {"from_node": "looper", "to_node": "looper"},  # self-loop
        ]
        config = {"services": {}}

        with pytest.raises(RouterGraphBuilderError, match="cycle"):
            _validate_manifest_security(nodes, edges, config)


class TestProvenanceHMAC:
    """Provenance HMAC signing for compiled graphs."""

    def test_compute_provenance_hmac(self):
        """HMAC computed from graph metadata."""
        from contextunity.router.cortex.compiler.provenance import (
            compute_provenance_hmac,
        )

        hmac_value = compute_provenance_hmac(
            graph_name="project:test:my_graph",
            node_names=["node_b", "node_a"],  # order shouldn't matter
            tenant_id="test-tenant",
            signing_key="test-key-123",
        )

        assert isinstance(hmac_value, str)
        assert len(hmac_value) > 0

    def test_same_inputs_produce_same_hmac(self):
        """Deterministic — same inputs → same HMAC."""
        from contextunity.router.cortex.compiler.provenance import (
            compute_provenance_hmac,
        )

        kwargs = {
            "graph_name": "graph1",
            "node_names": ["a", "b", "c"],
            "tenant_id": "t1",
            "signing_key": "key1",
        }
        assert compute_provenance_hmac(**kwargs) == compute_provenance_hmac(**kwargs)

    def test_different_nodes_produce_different_hmac(self):
        """Different node lists → different HMAC."""
        from contextunity.router.cortex.compiler.provenance import (
            compute_provenance_hmac,
        )

        base = {"graph_name": "g1", "tenant_id": "t1", "signing_key": "k1"}
        h1 = compute_provenance_hmac(node_names=["a", "b"], **base)
        h2 = compute_provenance_hmac(node_names=["a", "c"], **base)
        assert h1 != h2

    def test_node_order_does_not_matter(self):
        """Node names are sorted before HMAC — order irrelevant."""
        from contextunity.router.cortex.compiler.provenance import (
            compute_provenance_hmac,
        )

        base = {"graph_name": "g1", "tenant_id": "t1", "signing_key": "k1"}
        h1 = compute_provenance_hmac(node_names=["b", "a", "c"], **base)
        h2 = compute_provenance_hmac(node_names=["a", "b", "c"], **base)
        assert h1 == h2

    def test_verify_provenance_hmac_valid(self):
        """Valid HMAC passes verification."""
        from contextunity.router.cortex.compiler.provenance import (
            compute_provenance_hmac,
            verify_provenance_hmac,
        )

        hmac_val = compute_provenance_hmac(
            graph_name="g1",
            node_names=["a", "b"],
            tenant_id="t1",
            signing_key="k1",
        )
        assert verify_provenance_hmac(
            hmac_val,
            graph_name="g1",
            node_names=["a", "b"],
            tenant_id="t1",
            signing_key="k1",
        )

    def test_verify_provenance_hmac_tampered(self):
        """Tampered HMAC fails verification."""
        from contextunity.router.cortex.compiler.provenance import (
            verify_provenance_hmac,
        )

        assert not verify_provenance_hmac(
            "tampered-hmac-value",
            graph_name="g1",
            node_names=["a", "b"],
            tenant_id="t1",
            signing_key="k1",
        )
