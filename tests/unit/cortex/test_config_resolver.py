"""Tests for compiler config resolution — 3-level hierarchy.

Config priority: per-node → graph-wide → Router defaults.

``resolve_node_config`` accepts two shapes for *graph_config*:

- **Full manifest** (has ``nodes`` or ``edges``): graph-wide node defaults are
  read from ``graph_config["config"]``. The blob may carry graph-only keys
  (prompts, visualizer, …); only :class:`NodeConfig` fields are merged.
- **Flat dict** (tests / callers that already unwrapped graph defaults): the
  dict itself is treated as the graph-wide blob.
"""

from contextunity.router.cortex.compiler.node_config import NodeConfig


def _nc(**kwargs: object) -> NodeConfig:
    return NodeConfig.model_validate(dict(kwargs))


# ── resolve_node_config ───────────────────────────────────────────


class TestResolveNodeConfig:
    """Test 3-level config merge: per-node → graph-wide → Router defaults."""

    def test_per_node_overrides_graph_wide(self):
        """Per-node config takes priority over graph-wide defaults."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        node_spec = {
            "name": "generator",
            "type": "llm",
            "config": {"temperature": 0.9, "max_tokens": 4000},
        }
        graph_config = {"temperature": 0.3, "max_tokens": 2000, "timeout": 120}
        router_defaults = _nc(temperature=0.7, max_tokens=1000, timeout=300)

        result = resolve_node_config(node_spec, graph_config, router_defaults)

        assert result["temperature"] == 0.9  # per-node wins
        assert result["max_tokens"] == 4000  # per-node wins
        assert result["timeout"] == 120  # graph-wide wins over router default

    def test_graph_wide_overrides_router_defaults(self):
        """Graph-wide config overrides Router defaults when node omits key."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        node_spec = {"name": "classifier", "type": "llm", "config": {}}
        graph_config = {"temperature": 0.1, "timeout": 60}
        router_defaults = _nc(temperature=0.7, timeout=300, max_tokens=2000)

        result = resolve_node_config(node_spec, graph_config, router_defaults)

        assert result["temperature"] == 0.1  # graph-wide wins
        assert result["timeout"] == 60  # graph-wide wins
        assert result["max_tokens"] == 2000  # router default fills gap

    def test_router_defaults_fill_gaps(self):
        """Router defaults used when neither node nor graph specifies."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        node_spec = {"name": "simple", "type": "llm", "config": {}}
        graph_config = {}
        router_defaults = _nc(temperature=0.7, max_tokens=2000)

        result = resolve_node_config(node_spec, graph_config, router_defaults)

        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 2000

    def test_empty_all_levels(self):
        """No config at any level → empty dict."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        result = resolve_node_config(
            {"name": "bare", "type": "llm", "config": {}},
            {},
            NodeConfig(),
        )
        assert result == {}

    def test_node_config_not_dict_uses_empty(self):
        """If node has no 'config' key, treat as empty."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        node_spec = {"name": "no_config", "type": "llm"}
        graph_config = {"temperature": 0.5}
        router_defaults = NodeConfig()

        result = resolve_node_config(node_spec, graph_config, router_defaults)
        assert result["temperature"] == 0.5

    def test_preserves_state_routing_keys(self):
        """state_input_key and state_output_key survive the merge."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        node_spec = {
            "name": "routed",
            "type": "llm",
            "config": {
                "state_input_key": "user_query",
                "state_output_key": "analysis",
            },
        }
        result = resolve_node_config(node_spec, {}, NodeConfig())

        assert result["state_input_key"] == "user_query"
        assert result["state_output_key"] == "analysis"

    def test_full_manifest_merges_only_nested_config(self):
        """Topology manifest must not merge nodes/edges/max_retries into NodeConfig."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        graph_config = {
            "nodes": [],
            "edges": [],
            "max_retries": 99,
            "config": {"temperature": 0.2},
        }
        router_defaults = _nc(temperature=0.5, max_tokens=1000)

        result = resolve_node_config(
            {"name": "gen", "type": "llm", "config": {}},
            graph_config,
            router_defaults,
        )

        assert result["temperature"] == 0.2
        assert result["max_tokens"] == 1000
        assert "max_retries" not in result
        assert "nodes" not in result

    def test_graph_wide_blob_with_graph_only_keys_ignored(self):
        """Graph-only keys in the blob (prompts, visualizer, …) don't enter NodeConfig."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        graph_config = {
            "nodes": [],
            "edges": [],
            "config": {
                "temperature": 0.15,
                "planner_prompt": "graph-only blob",
                "visualizer_prompt": "also graph-only",
                "visualizer_sub_prompts": {"chart": "…"},
                "node_tool_bindings": {"foo": "bar"},
            },
        }
        router_defaults = _nc(temperature=0.5, max_tokens=1000)

        result = resolve_node_config(
            {"name": "gen", "type": "llm", "config": {}},
            graph_config,
            router_defaults,
        )

        assert result["temperature"] == 0.15
        assert result["max_tokens"] == 1000
        assert "planner_prompt" not in result
        assert "visualizer_prompt" not in result
        assert "node_tool_bindings" not in result

    def test_project_manifest_projection_uses_nested_config_without_topology_keys(self):
        """Projected manifests without nodes/edges still read node defaults from config."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_node_config,
        )

        graph_config = {
            "max_retries": 2,
            "config": {
                "temperature": 0.15,
                "max_tokens": 4096,
                "planner_prompt": "graph-only blob",
            },
        }
        router_defaults = _nc(temperature=0.5, max_tokens=1000)

        result = resolve_node_config(
            {"name": "gen", "type": "llm", "config": {}},
            graph_config,
            router_defaults,
        )

        assert result["temperature"] == 0.15
        assert result["max_tokens"] == 4096
        assert "max_retries" not in result
        assert "planner_prompt" not in result


# ── resolve_model ─────────────────────────────────────────────────


class TestResolveModel:
    """Test model resolution fallback chain."""

    def test_node_model_wins(self):
        """Model specified on node takes top priority."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model,
        )

        result = resolve_model(
            node_spec={"model": "anthropic/claude-sonnet-4-20250514"},
            graph_config={"default_model": "openai/gpt-5-mini"},
            router_defaults=_nc(default_model="openai/gpt-4o"),
        )
        assert result == "anthropic/claude-sonnet-4-20250514"

    def test_graph_default_model_fallback(self):
        """Falls back to graph-wide default_model when node omits model."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model,
        )

        result = resolve_model(
            node_spec={"name": "no_model"},
            graph_config={"default_model": "openai/gpt-5-mini"},
            router_defaults=NodeConfig(),
        )
        assert result == "openai/gpt-5-mini"

    def test_router_default_model_fallback(self):
        """Falls back to Router default when graph also omits model."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model,
        )

        result = resolve_model(
            node_spec={},
            graph_config={},
            router_defaults=_nc(default_model="openai/gpt-4o"),
        )
        assert result == "openai/gpt-4o"

    def test_no_model_anywhere_returns_empty(self):
        """No model at any level → empty string (caller handles)."""
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model,
        )

        result = resolve_model(node_spec={}, graph_config={}, router_defaults=NodeConfig())
        assert result == ""


# ── resolve_model_secret_ref ──────────────────────────────────────


class TestResolveModelSecretRef:
    """Test secret ref resolution chain."""

    def test_node_secret_ref_wins(self):
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model_secret_ref,
        )

        result = resolve_model_secret_ref(
            node_spec={"model_secret_ref": "NODE_KEY"},
            graph_config={"default_model_secret_ref": "GRAPH_KEY"},
            router_defaults=_nc(default_model_secret_ref="ROUTER_KEY"),
        )
        assert result == "NODE_KEY"

    def test_graph_default_secret_ref_fallback(self):
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model_secret_ref,
        )

        result = resolve_model_secret_ref(
            node_spec={},
            graph_config={"default_model_secret_ref": "GRAPH_KEY"},
            router_defaults=NodeConfig(),
        )
        assert result == "GRAPH_KEY"

    def test_router_default_secret_ref_fallback(self):
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model_secret_ref,
        )

        result = resolve_model_secret_ref(
            node_spec={},
            graph_config={},
            router_defaults=_nc(default_model_secret_ref="ROUTER_KEY"),
        )
        assert result == "ROUTER_KEY"

    def test_no_secret_ref_returns_none(self):
        from contextunity.router.cortex.compiler.config_resolver import (
            resolve_model_secret_ref,
        )

        result = resolve_model_secret_ref(
            node_spec={}, graph_config={}, router_defaults=NodeConfig()
        )
        assert result is None


# ── get_router_defaults ───────────────────────────────────────────


class TestGetRouterDefaults:
    """Test extraction of Router-level defaults from RouterConfig."""

    def test_returns_nodeconfig(self):
        """Should return a :class:`NodeConfig` instance."""
        from contextunity.router.cortex.compiler.config_resolver import (
            get_router_defaults,
        )

        defaults = get_router_defaults()
        assert isinstance(defaults, NodeConfig)
