"""Custom Dynamic Graph Builder — Declarative Topology Compiler for LangGraph.

This module compiles declarative representations of nodes and edges (defined in project
manifests or templates) into executable LangGraph `StateGraph` instances. It integrates configuration
resolution, dependency checking, memory injection, and strict security validation gates.

Security by Construction Invariants:
    1. Pre-Compilation Validation: The configuration is validated via `validate_manifest_security()`
       before any node is built or added to the graph.
    2. Sandbox wrapping: Every node execution callback is unconditionally wrapped with `make_secure_node()`.
    3. Loop/Cycle Detection: Graph node count is limited to prevent graph exhaustion or execution loops.
    4. Access Control Boundaries: Platform tools are validated against tenant-level and service-level
       permissions to enforce the principle of least privilege.
"""

from collections.abc import Hashable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict
from langgraph.graph import END, START, StateGraph

from contextunity.router.core.exceptions import RouterGraphBuilderError
from contextunity.router.cortex.compiler.node_config import NodeConfig
from contextunity.router.cortex.compiler.node_factory import add_compiled_node
from contextunity.router.cortex.compiler.state import create_condition
from contextunity.router.cortex.compiler.topology import (
    analyze_topology,
    auto_complete_entry_exit_edges,
)
from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    ProjectManifest,
    TemplateBuildOverrides,
)
from contextunity.router.cortex.compiler.validation import (
    MAX_NODES_DEFAULT,
    validate_manifest_security,
)
from contextunity.router.cortex.types import CortexGraph, GraphState

logger = get_contextunit_logger(__name__)
type CompilerGraph = StateGraph[GraphState, None, GraphState, GraphState]

# Re-export for existing compiler tests that validate the security gate directly.
_validate_manifest_security = validate_manifest_security


def build_local_graph(
    nodes: list[CompilerNodeSpec],
    edges: list[CompilerEdgeSpec],
    config: ProjectManifest,
    *,
    max_nodes: int = MAX_NODES_DEFAULT,
    router_defaults: NodeConfig | None = None,
) -> CompilerGraph:
    """Build an uncompiled LangGraph StateGraph from manifest node and edge specifications.

    Resolves the provided graph topology, injects auxiliary system/memory nodes, performs
    three-tier configuration resolution (node-level, graph-level, router defaults), validates
    integrity, and compiles the final StateGraph.

    Args:
        nodes: List of declarative node specifications containing types, tool list, and parameters.
        edges: List of edge specifications mapping node transition routes and conditional logic.
        config: Combined project configuration manifest containing model, service, and security settings.
        max_nodes: Hard limit on the maximum number of nodes permitted in the graph.
        router_defaults: Default configurations for models, fallback providers, and time-to-live settings.

    Returns:
        StateGraph: The uncompiled LangGraph topology ready for compilation.

    Raises:
        RouterGraphBuilderError: If the manifest fails pre-compilation security checks, contains
            dangling edges, invalid node names, or references missing services.
    """
    # Auto-complete entry/exit edges for local manifests when omitted.
    edges = auto_complete_entry_exit_edges(nodes, edges)

    # ── Pre-compilation security validation ──
    _validate_manifest_security(nodes, edges, config, max_nodes=max_nodes)

    # ── Resolve Router defaults ──
    if router_defaults is None:
        from contextunity.router.cortex.compiler.config_resolver import (
            get_router_defaults,
        )

        router_defaults = get_router_defaults()

    # ── Brain Phase C: Memory injection ──
    from contextunity.router.cortex.compiler.memory_injection import (
        inject_memory_nodes,
    )

    nodes, edges = inject_memory_nodes(nodes, edges, config)

    # ── Phase 4: Trace injection (observability) ──
    # Note: _trace node injection is deprecated. Traces are now logged
    # automatically by StreamAgent at the end of the execution via BrainClient.log_trace.
    # nodes, edges = inject_trace_node(nodes, edges, config)

    logger.debug("🔧 Compiling Dynamic Local Graph (%d nodes)...", len(nodes))
    graph = CortexGraph(GraphState)

    entry_nodes, json_required_nodes = analyze_topology(edges)

    manifest = config

    # 1. Add Nodes — every node gets make_secure_node() unconditionally
    for node_spec in nodes:
        add_compiled_node(
            graph,
            node_spec,
            manifest=manifest,
            router_defaults=router_defaults,
            entry_nodes=entry_nodes,
            json_required_nodes=json_required_nodes,
        )

    # 2. Add Edges
    for edge in edges:
        from_node = edge.get("from_node")
        to_node = edge.get("to_node")
        cond_key = edge.get("condition_key")
        cond_map = edge.get("condition_map")

        if not from_node:
            raise RouterGraphBuilderError(message="Edge missing 'from_node' field.")

        if from_node == "__start__":
            from_node = START

        if cond_key and cond_map:
            logger.debug("Adding custom conditional edge from %s (key: %s)", from_node, cond_key)

            # map internal __end__ and __start__ representation
            mapped_cond_map: dict[str, str] = {}
            for k, v in cond_map.items():
                if v == "__end__":
                    mapped_cond_map[k] = END
                elif v == "__start__":
                    mapped_cond_map[k] = START
                else:
                    mapped_cond_map[k] = v

            path_map: dict[Hashable, str] = {
                condition: target for condition, target in mapped_cond_map.items()
            }
            _ = graph.add_conditional_edges(
                from_node,
                create_condition(cond_key, mapped_cond_map),
                path_map,
            )
        elif to_node:
            if to_node == "__end__":
                to_node = END
            logger.debug("Adding custom edge: %s -> %s", from_node, to_node)
            _ = graph.add_edge(from_node, to_node)
        else:
            raise RouterGraphBuilderError(
                message=f"Edge from '{from_node}' missing both 'to_node' and 'condition_key'."
            )

    return graph


def build_from_template(
    template_name: str,
    overrides: TemplateBuildOverrides | None = None,
    config: ProjectManifest | None = None,
    *,
    max_nodes: int = MAX_NODES_DEFAULT,
    router_defaults: NodeConfig | None = None,
) -> CompilerGraph:
    """Build an uncompiled StateGraph from a named declarative template with overrides.

    This function coordinates the complete template expansion pipeline:
        1. `load_template(name)` — Loads the YAML template configuration and validates schemas.
        2. `merge_overrides(tpl, overrides)` — Applies node-level parameter or tool list overrides.
        3. Extract nodes and edges from the template.
        4. Merges template defaults (models, secret references) and topology config into the project config.
        5. Invokes `build_local_graph` to produce the fully secure, wrapped `StateGraph`.

    This is the primary developer-facing graph compilation entrypoint. When a tenant manifest
    declares a predefined template (such as `gardener` or `enricher`), this builder parses,
    validates, and links all nodes and transitions.

    Args:
        template_name: The name or identifier of the template to load (e.g., 'gardener',
            'retrieval_augmented'). Semantically supports alias redirection for deprecated
            keys (`rag_retrieval`, `sql_analytics`).
        overrides: Optional mapping of node-specific parameter overrides to customize template execution.
        config: Optional project/tenant configuration manifest to merge with the template defaults.
        max_nodes: Hard ceiling limit for total node count in the final compiled graph.
        router_defaults: Default configurations for models and parameters.

    Returns:
        StateGraph: The uncompiled LangGraph topology representing the template config.

    Raises:
        ConfigurationError: If the template is not found, contains schema violations, or
            defines invalid node/parameter overrides.
        RouterGraphBuilderError: If the constructed topology fails security/integrity constraints.
    """
    from contextunity.router.cortex.compiler.template_loader import (
        load_template,
        merge_overrides,
    )

    # Apply deprecation aliases (Phase 6.8)
    if template_name in ("rag_retrieval", "sql_analytics"):
        logger.warning(
            (
                "DEPRECATED: Template '%s' is deprecated. "
                "It will seamlessly alias to 'retrieval_augmented'. "
                "Please migrate to 'retrieval_augmented' with data_sources."
            ),
            template_name,
        )
        # Ensure config provides legacy fallback data source
        config = config or {}
        inner_config_raw = config.get("config")
        inner_config = dict(inner_config_raw) if is_object_dict(inner_config_raw) else {}
        if "data_sources" not in inner_config:
            if template_name == "rag_retrieval":
                inner_config["data_sources"] = [{"type": "vector", "binding": "default_vector"}]
            else:
                inner_config["data_sources"] = [{"type": "sql", "binding": "default_sql"}]
            _ = config.setdefault("config", inner_config)
        template_name = "retrieval_augmented"

    # Load + validate template
    template = load_template(template_name)

    # Apply consumer overrides
    if overrides:
        template = merge_overrides(template, overrides)

    # Convert Pydantic models → TypedDicts for build_local_graph
    nodes: list[CompilerNodeSpec] = [node.to_spec() for node in template.nodes]
    edges: list[CompilerEdgeSpec] = [edge.to_spec() for edge in template.edges]

    # Merge template defaults into config
    merged_config: ProjectManifest = {**config} if config else {}
    if template.defaults.model:
        _ = merged_config.setdefault("model", template.defaults.model)
    if template.defaults.model_secret_ref:
        _ = merged_config.setdefault("model_secret_ref", template.defaults.model_secret_ref)

    # Merge template graph-level configuration (e.g. max_retries for cycle detection).
    if "max_retries" not in merged_config:
        merged_config["max_retries"] = template.config.max_retries
    if "timeout" not in merged_config:
        merged_config["timeout"] = template.config.timeout
    if "data_sources" not in merged_config:
        merged_config["data_sources"] = [
            {
                "type": data_source.type,
                "binding": data_source.binding,
                "description": data_source.description,
                "config": dict(data_source.config),
            }
            for data_source in template.config.data_sources
        ]
    if "pipeline" not in merged_config:
        merged_config["pipeline"] = {
            "memory": template.config.pipeline.memory,
            "memory_depth": "standard",
            "reflection": template.config.pipeline.reflection,
            "verification": template.config.pipeline.verification,
            "visualization": template.config.pipeline.visualization,
            "suggestions": template.config.pipeline.suggestions,
        }

    logger.info(
        "📋 Building graph from template '%s' v%s (%d nodes, %d edges)",
        template.name,
        template.version,
        len(nodes),
        len(edges),
    )

    return build_local_graph(
        nodes,
        edges,
        merged_config,
        max_nodes=max_nodes,
        router_defaults=router_defaults,
    )
