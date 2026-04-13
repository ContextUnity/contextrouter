"""Custom Dynamic Graph Builder.

Constructs LangGraph StateGraph directly from manifest declarative nodes and edges.
Provides generic State and generic node executors (LLM, Tool).

Security by Construction:
- Every node is unconditionally wrapped with make_secure_node()
- Reserved node names are rejected at compile time
- Node count is limited to prevent graph bombs
- Node names are validated to prevent provenance forgery
"""

import logging
import operator
import re
from typing import Annotated, Any, TypedDict

from contextunity.core.exceptions import GraphBuilderError, SecurityError
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)

# ── Security Constants ──────────────────────────────────────────────

# Reserved names that could be used for provenance forgery or confusion.
RESERVED_NODE_NAMES: frozenset[str] = frozenset(
    {
        "system",
        "admin",
        "root",
        "shield",
        "brain",
        "zero",
        "worker",
        "router",
        "dispatcher",
        "kernel",
        "__start__",
        "__end__",
    }
)

# Max nodes per compiled graph. Prevents graph bombs (exponential fan-out).
# Projects can request a higher limit via manifest policy, but this is the hard ceiling.
MAX_NODES_DEFAULT: int = 50

# Node name format: lowercase alphanumeric + underscores only.
_NODE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class CustomGraphState(TypedDict):
    """Generic State for Custom Graphs."""

    tenant_id: str
    messages: Annotated[list[Any], operator.add]
    intermediate_results: dict[str, Any]
    final_output: dict[str, Any]

    # Specific keys for content generation (like PinkPony)
    raw_items: list[dict[str, Any]]
    posts: Annotated[list[dict[str, Any]], operator.add]


def make_llm_node(node_spec: dict, config: dict):
    """Create a generic LangGraph node for an LLM."""
    node_name = node_spec.get("name")
    model_name = node_spec.get("model")
    prompt_ref = node_spec.get("prompt_ref")

    async def llm_executor(state: CustomGraphState) -> dict[str, Any]:
        logger.info("Executing Custom LLM Node: %s", node_name)
        state.get("tenant_id", "default")

        # Load prompt from config if provided via ArtifactGenerator
        config.get(f"{prompt_ref}_prompt") if prompt_ref else "Default custom prompt"

        # In a real dynamic graph, we'd use contextual information
        # Here we just mock generating a post for demonstration
        result_post = {
            "source": node_name,
            "content": f"Generated result from {node_name} using {model_name}",
        }

        # Return state update using operator.add aggregator
        return {"posts": [result_post]}

    return llm_executor


def make_tool_node(node_spec: dict, config: dict):
    """Create a generic LangGraph node for a Tool (like federated store_news_results)."""
    node_name = node_spec.get("name")
    tool_binding = node_spec.get("tool_binding")

    async def tool_executor(state: CustomGraphState) -> dict[str, Any]:
        logger.info("Executing Custom Tool Node: %s -> %s", node_name, tool_binding)
        tenant_id = state.get("tenant_id", "default")
        posts = state.get("posts", [])

        if not posts:
            logger.warning("[%s] No posts to process", node_name)
            return {}

        try:
            from contextunity.core.sdk.clients.router import RouterClient

            client = RouterClient()
            logger.info("[%s] Dispatching %d posts to %s...", node_name, len(posts), tool_binding)

            # Execute tool bound to this project
            await client.execute_tool(
                tool_name=tool_binding, args={"results": posts}, target_project=tenant_id
            )
        except Exception as e:
            logger.error("[%s] Tool execution failed: %s", node_name, e)

        return {}

    return tool_executor


def _create_condition(condition_key: str):
    def _condition_func(state: CustomGraphState) -> str:
        val = state.get("intermediate_results", {}).get(condition_key)
        if not val:
            val = state.get(condition_key)
        return str(val) if val else "default"

    return _condition_func


# ── Security Validation ─────────────────────────────────────────────


def _validate_manifest_security(
    nodes: list[dict],
    *,
    max_nodes: int = MAX_NODES_DEFAULT,
) -> None:
    """Validate manifest security invariants before compilation.

    Fail-closed: any violation raises immediately, graph is never compiled.
    """
    # M7.1: Node count limit — prevents graph bombs
    if len(nodes) > max_nodes:
        raise GraphBuilderError(
            message=(
                f"Graph exceeds max_nodes limit: {len(nodes)} > {max_nodes}. "
                "Reduce node count or request a higher limit via project policy."
            ),
        )

    seen_names: set[str] = set()
    for node_spec in nodes:
        name = node_spec.get("name")
        if not name:
            raise GraphBuilderError(message="Node is missing required 'name' field.")

        # M1.2: Reserved name check — prevents provenance forgery
        if name.lower() in RESERVED_NODE_NAMES:
            raise SecurityError(
                message=(
                    f"Node name '{name}' is reserved and cannot be used in manifests. "
                    f"Reserved names: {sorted(RESERVED_NODE_NAMES)}"
                ),
                node_name=name,
            )

        # Name format validation — prevents injection via special characters
        if not _NODE_NAME_RE.match(name):
            raise GraphBuilderError(
                message=(
                    f"Invalid node name '{name}'. "
                    "Node names must match ^[a-z][a-z0-9_]{{0,63}}$ "
                    "(lowercase, alphanumeric, underscores, max 64 chars)."
                ),
            )

        # Duplicate name check
        if name in seen_names:
            raise GraphBuilderError(
                message=f"Duplicate node name '{name}' in manifest.",
            )
        seen_names.add(name)

        # M3.1: model_secret_ref format — no path traversal
        secret_ref = node_spec.get("model_secret_ref")
        if secret_ref and not re.match(r"^[a-zA-Z0-9_-]+$", secret_ref):
            raise SecurityError(
                message=(
                    f"Invalid model_secret_ref '{secret_ref}' on node '{name}'. "
                    "Secret references must be alphanumeric with underscores/hyphens only. "
                    "Path separators are forbidden to prevent traversal attacks."
                ),
                node_name=name,
            )


def build_custom_graph(
    nodes: list[dict],
    edges: list[dict],
    config: dict,
    *,
    max_nodes: int = MAX_NODES_DEFAULT,
) -> object:
    """Build LangGraph from manifest nodes and edges.

    Security by Construction:
    - _validate_manifest_security() runs BEFORE any node is created
    - Every node is wrapped with make_secure_node() — no exceptions
    - Reserved names, node limits, and secret ref formats are enforced
    """
    # ── Pre-compilation security validation ──
    _validate_manifest_security(nodes, max_nodes=max_nodes)

    logger.info("🔧 Compiling Dynamic Custom Graph (%d nodes)...", len(nodes))
    graph = StateGraph(CustomGraphState)

    # 1. Add Nodes — every node gets make_secure_node() unconditionally
    for node_spec in nodes:
        name = node_spec.get("name")
        type_ = node_spec.get("type", "llm")
        logger.info("Adding custom node: %s (type: %s)", name, type_)

        from contextunity.router.cortex.graphs.secure_node import make_secure_node

        model_ref = node_spec.get("model_secret_ref")
        pii = node_spec.get("pii_masking", False)
        tools = node_spec.get("tools", [])
        if type_ == "tool" and node_spec.get("tool_binding"):
            tools.append(node_spec.get("tool_binding"))

        if type_ == "llm":
            node_func = make_secure_node(
                name,
                make_llm_node(node_spec, config),
                pii_masking=pii,
                model_secret_ref=model_ref,
                tools=tools,
            )
            graph.add_node(name, node_func)
        elif type_ == "tool":
            node_func = make_secure_node(
                name,
                make_tool_node(node_spec, config),
                pii_masking=pii,
                model_secret_ref=model_ref,
                tools=tools,
            )
            graph.add_node(name, node_func)
        else:
            raise GraphBuilderError(
                message=f"Unsupported node type '{type_}' on node '{name}'. "
                f"Supported types: llm, tool."
            )

    # 2. Add Edges
    for edge in edges:
        from_node = edge.get("from")
        to_node = edge.get("to")
        cond_key = edge.get("condition_key")
        cond_map = edge.get("condition_map")

        if from_node == "__start__":
            from_node = START

        if cond_key and cond_map:
            logger.info("Adding custom conditional edge from %s (key: %s)", from_node, cond_key)

            # map internal __end__ and __start__ representation
            mapped_cond_map = {}
            for k, v in cond_map.items():
                if v == "__end__":
                    mapped_cond_map[k] = END
                elif v == "__start__":
                    mapped_cond_map[k] = START
                else:
                    mapped_cond_map[k] = v

            graph.add_conditional_edges(from_node, _create_condition(cond_key), mapped_cond_map)
        elif to_node:
            if to_node == "__end__":
                to_node = END
            logger.info("Adding custom edge: %s -> %s", from_node, to_node)
            graph.add_edge(from_node, to_node)
        else:
            raise GraphBuilderError(
                message=f"Edge from '{from_node}' missing both 'to' and 'condition_key'."
            )

    return graph.compile()
