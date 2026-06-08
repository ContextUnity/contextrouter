"""Manifest Security Validation for the Graph Compiler.

Security by Construction:
- Reserved node names are rejected at compile time
- Node count is limited to prevent graph bombs
- Node names are validated to prevent provenance forgery
- Edge integrity: dangling from:/to: references rejected
- Service dependencies: platform-bound tools require their service enabled
- Tool bindings: federated tools must use the federated: namespace; bare bindings are platform shorthand
- Cycle detection: back-edges require config.max_retries
"""

import re

from contextunity.core.exceptions import SecurityError
from contextunity.core.manifest.helpers import parse_tool_ref
from contextunity.core.types import is_object_dict, is_object_list

from contextunity.router.core.exceptions import RouterGraphBuilderError
from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    ProjectManifest,
    resolve_manifest_max_retries,
)

# Reserved names that could be used for provenance forgery or confusion.
RESERVED_NODE_NAMES: frozenset[str] = frozenset(
    {
        "system",
        "admin",
        "root",
        "shield",
        "brain",
        "privacy",
        "worker",
        "router",
        "dispatcher",
        "kernel",
        "__start__",
        "__end__",
        # Platform-injected nodes (memory_injection.py)
        "_memory_load",
        "_memory_save",
        # Observability (Phase 4 — auto-injected trace node)
        "_trace",
    }
)

# Max nodes per compiled graph. Prevents graph bombs (exponential fan-out).
MAX_NODES_DEFAULT: int = 50

# Node name format: lowercase alphanumeric + underscores only.
_NODE_NAME_RE = re.compile(r"^[_a-z][a-z0-9_]{0,63}$")

# Service prefix → required service key in config.services
_SERVICE_PREFIX_MAP: dict[str, str] = {
    "brain_": "brain",
    "shield_": "shield",
    "worker_": "worker",
}

# Self-hosted prefixes — tools that run inside the Router process itself.
# No external service dependency required.
_SELF_HOSTED_PREFIXES: frozenset[str] = frozenset({"router_", "language_"})


def validate_manifest_security(
    nodes: list[CompilerNodeSpec],
    edges: list[CompilerEdgeSpec],
    config: ProjectManifest,
    *,
    max_nodes: int = MAX_NODES_DEFAULT,
) -> None:
    """Validate manifest security invariants before compilation."""
    # Node count limit — prevents graph bombs
    if len(nodes) > max_nodes:
        raise RouterGraphBuilderError(
            message=(
                f"Graph exceeds max_nodes limit: {len(nodes)} > {max_nodes}. "
                "Reduce node count or request a higher limit via project policy."
            ),
        )

    seen_names: set[str] = set()
    for node_spec in nodes:
        name = node_spec.get("name")
        if not name:
            raise RouterGraphBuilderError(message="Node is missing required 'name' field.")

        # Reserved name check — prevents provenance forgery
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
            raise RouterGraphBuilderError(
                message=(
                    f"Invalid node name '{name}'. "
                    "Node names must match ^[_a-z][a-z0-9_]{{0,63}}$ "
                    "(lowercase, alphanumeric, underscores, max 64 chars)."
                ),
            )

        # Duplicate name check
        if name in seen_names:
            raise RouterGraphBuilderError(
                message=f"Duplicate node name '{name}' in manifest.",
            )
        seen_names.add(name)

        # model_secret_ref format — no path traversal
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

        node_type = node_spec.get("type", "llm")
        if node_type not in {"llm", "agent", "tool"}:
            raise RouterGraphBuilderError(
                message=(
                    f"Unsupported node type '{node_type}' on node '{name}'. "
                    "Supported types: llm, agent, tool. Platform routing is selected "
                    "with tool_binding: platform:<name> or bare platform shorthand."
                ),
            )

        if node_type in {"llm", "agent"}:
            _validate_prompt_ref_resolved(node_spec, name, config)

        if node_type == "agent":
            raw_tools: object = node_spec.get("tools") or []
            if not is_object_list(raw_tools):
                raise RouterGraphBuilderError(
                    message=f"Agent node '{name}' has invalid tools list: expected list[str].",
                )
            for tool_ref_obj in raw_tools:
                from contextunity.core.manifest.helpers import TOOL_REF_RE

                if not isinstance(tool_ref_obj, str):
                    raise RouterGraphBuilderError(
                        message=(
                            f"Agent node '{name}' has non-string tool reference: {tool_ref_obj!r}"
                        ),
                    )
                if not TOOL_REF_RE.match(tool_ref_obj):
                    raise RouterGraphBuilderError(
                        message=(
                            f"Agent node '{name}' tool ref '{tool_ref_obj}' is invalid. "
                            "Expected 'platform:<name>' or 'federated:<name>'."
                        ),
                    )
            raw_toolkits: object = node_spec.get("toolkits") or []
            if raw_toolkits:
                raise RouterGraphBuilderError(
                    message=(
                        f"Agent node '{name}' declares toolkits. "
                        "SDK bundles must expand toolkits to explicit tools before registration."
                    ),
                )

        # Tool node binding routes by namespace. Bare bindings are platform shorthand.
        if node_type == "tool":
            tool_binding = node_spec.get("tool_binding", "")
            _validate_tool_binding(tool_binding, name)
            kind, _tool_name = parse_tool_ref(tool_binding)
            if kind == "platform":
                _validate_service_dependency(tool_binding, name, config)

    # Edge integrity — every from:/to: references existing node or __start__/__end__
    _validate_edge_integrity(edges, seen_names)

    # Cycle detection — back-edges require config.max_retries
    _validate_no_unguarded_cycles(edges, seen_names, config)


def _validate_service_dependency(
    tool_binding: str, node_name: str, config: ProjectManifest
) -> None:
    """Reject platform-bound tools whose target service is not enabled in the manifest."""
    kind, tool_name = parse_tool_ref(tool_binding)
    if kind and kind != "platform":
        raise RouterGraphBuilderError(
            message=(
                f"Node '{node_name}' uses non-platform tool binding '{tool_binding}' "
                "on a platform node. Use type='tool' for federated bindings."
            ),
        )
    binding_name = tool_name or tool_binding

    # Self-hosted tools — no external service dependency
    for prefix in _SELF_HOSTED_PREFIXES:
        if binding_name.startswith(prefix):
            return

    services_config = config.get("services", {})

    for prefix, service_key in _SERVICE_PREFIX_MAP.items():
        if binding_name.startswith(prefix):
            service_conf = services_config.get(service_key, {})
            if not is_object_dict(service_conf) or service_conf.get("enabled") is not True:
                raise RouterGraphBuilderError(
                    message=(
                        f"Node '{node_name}' uses platform tool '{tool_binding}' which requires "
                        f"service '{service_key}' to be enabled. Add "
                        f"'services.{service_key}.enabled: true' to project manifest."
                    ),
                )
            return

    # Unknown prefix — no service match
    raise RouterGraphBuilderError(
        message=(
            f"Unknown platform tool prefix in '{tool_binding}' on node '{node_name}'. "
            f"Expected prefixes: {sorted([*_SERVICE_PREFIX_MAP.keys(), *_SELF_HOSTED_PREFIXES])}"
        ),
    )


def _validate_prompt_ref_resolved(
    node_spec: CompilerNodeSpec,
    node_name: str,
    config: ProjectManifest,
) -> None:
    """Reject nodes whose declared prompt_ref was not resolved into graph config."""
    prompt_ref = node_spec.get("prompt_ref")
    if not isinstance(prompt_ref, str) or not prompt_ref.strip():
        return

    inner_config = config.get("config", config)
    resolved_prompt: object | None = None
    if is_object_dict(inner_config):
        resolved_prompt = inner_config.get(f"{node_name}_prompt")

    if not isinstance(resolved_prompt, str) or not resolved_prompt.strip():
        raise RouterGraphBuilderError(
            message=(
                f"Node '{node_name}' declares prompt_ref '{prompt_ref}' but no resolved "
                f"'{node_name}_prompt' was found in graph config. Ensure the project SDK "
                "resolves prompt_ref before registration and the referenced prompt file exists."
            ),
            node_name=node_name,
            prompt_ref=prompt_ref,
            prompt_key=f"{node_name}_prompt",
        )


def _validate_tool_binding(tool_binding: str, node_name: str) -> None:
    """Reject tool nodes with empty, non-string, or malformed bindings."""
    if not tool_binding:
        raise RouterGraphBuilderError(
            message=(
                f"Node '{node_name}' has invalid tool_binding. "
                "Expected '<platform_tool>', 'platform:<name>', or 'federated:<name>'."
            ),
        )
    from contextunity.core.manifest.helpers import TOOL_BINDING_RE

    if not TOOL_BINDING_RE.match(tool_binding.strip()):
        raise RouterGraphBuilderError(
            message=(
                f"Node '{node_name}' tool_binding '{tool_binding}' is invalid. "
                "Expected '<platform_tool>', 'platform:<name>', or 'federated:<name>'."
            ),
        )


def _validate_edge_integrity(edges: list[CompilerEdgeSpec], node_names: set[str]) -> None:
    """Reject edges whose ``from_node`` or ``to_node`` reference non-existent nodes."""
    valid_refs = node_names | {"__start__", "__end__"}

    for edge in edges:
        from_node = edge.get("from_node")
        to_node = edge.get("to_node")
        cond_map = edge.get("condition_map")

        # Validate 'from' — must exist
        if from_node and from_node not in valid_refs:
            raise RouterGraphBuilderError(
                message=(
                    f"Edge references non-existent source node '{from_node}'. "
                    f"Valid nodes: {sorted(valid_refs)}"
                ),
            )

        # Validate 'to' — must exist (unless conditional edge)
        if to_node and to_node not in valid_refs:
            raise RouterGraphBuilderError(
                message=(
                    f"Edge references non-existent target node '{to_node}'. "
                    f"Valid nodes: {sorted(valid_refs)}"
                ),
            )

        # Validate condition_map targets
        if cond_map:
            for cond_val, target in cond_map.items():
                if target not in valid_refs:
                    raise RouterGraphBuilderError(
                        message=(
                            f"Conditional edge target '{target}' (condition='{cond_val}') "
                            f"references non-existent node. "
                            f"Valid nodes: {sorted(valid_refs)}"
                        ),
                    )


def _validate_no_unguarded_cycles(
    edges: list[CompilerEdgeSpec],
    node_names: set[str],
    config: ProjectManifest,
) -> None:
    """Detect graph cycles via DFS; require ``max_retries`` when back-edges are present."""
    # Build adjacency list from edges (only user nodes, skip __start__/__end__)
    adj: dict[str, list[str]] = {name: [] for name in node_names}

    for edge in edges:
        from_node = edge.get("from_node", "")
        to_node = edge.get("to_node")
        cond_map = edge.get("condition_map")

        # Skip edges from/to __start__/__end__ — they're not cycles
        if from_node in ("__start__", "__end__") or from_node not in node_names:
            continue

        if to_node and to_node in node_names:
            adj[from_node].append(to_node)

        if cond_map:
            for target in cond_map.values():
                if target in node_names:
                    adj[from_node].append(target)

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {name: WHITE for name in node_names}
    has_cycle = False

    def dfs(node: str) -> bool:
        """Return ``True`` when a back-edge (GRAY → GRAY) is found, marking a cycle."""
        nonlocal has_cycle
        color[node] = GRAY
        for neighbor in adj[node]:
            if color[neighbor] == GRAY:
                # Back-edge found → cycle
                has_cycle = True
                return True
            if color[neighbor] == WHITE:
                if dfs(neighbor):
                    return True
        color[node] = BLACK
        return False

    for node in node_names:
        if color[node] == WHITE:
            _ = dfs(node)

    if has_cycle:
        max_retries = resolve_manifest_max_retries(config)
        if max_retries is None:
            raise RouterGraphBuilderError(
                message=(
                    "Graph contains a cycle (back-edge detected) but "
                    "'config.max_retries' is not set. Cycles require "
                    "max_retries to prevent infinite loops. "
                    "Add 'config: { max_retries: N }' to the graph manifest."
                ),
            )


__all__ = [
    "validate_manifest_security",
    "RESERVED_NODE_NAMES",
    "MAX_NODES_DEFAULT",
]
