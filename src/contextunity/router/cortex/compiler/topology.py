"""Topology helpers — edge wiring, conditional routing, and tool-ref resolution for compiled graphs."""

from __future__ import annotations

from contextunity.core.manifest.helpers import parse_tool_ref

from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    TopologyInfo,
)

# Backward-compatible alias — compiler internals use this name.
parse_binding = parse_tool_ref


def normalize_tool_ref(ref: str) -> str:
    """Strip the kind prefix from federated/platform tool refs, returning the bare name."""
    kind, name = parse_tool_ref(ref)
    return name if kind in {"federated", "platform"} else ref


def auto_complete_entry_exit_edges(
    nodes: list[CompilerNodeSpec],
    edges: list[CompilerEdgeSpec],
) -> list[CompilerEdgeSpec]:
    """Auto-inject ``__start__`` → first-node and terminal-nodes → ``__end__`` edges when omitted."""
    if not nodes:
        return edges
    updated_edges: list[CompilerEdgeSpec] = list(edges)
    node_names = [str(node.get("name")) for node in nodes if node.get("name")]
    if not node_names:
        return updated_edges

    has_start = any(edge.get("from_node") == "__start__" for edge in updated_edges)
    if not has_start:
        updated_edges.insert(0, {"from_node": "__start__", "to_node": node_names[0]})

    has_end = any(edge.get("to_node") == "__end__" for edge in updated_edges)
    if has_end:
        return updated_edges

    outgoing: set[str] = set()
    for edge in updated_edges:
        from_node = edge.get("from_node")
        if isinstance(from_node, str):
            outgoing.add(from_node)

    terminals = [name for name in node_names if name not in outgoing]
    if not terminals:
        terminals = [node_names[-1]]
    for terminal in terminals:
        updated_edges.append({"from_node": terminal, "to_node": "__end__"})
    return updated_edges


def analyze_topology(edges: list[CompilerEdgeSpec]) -> TopologyInfo:
    """Return entry nodes and nodes whose conditional edges require json."""
    entry_nodes: set[str] = set()
    json_required_nodes: set[str] = set()
    for edge in edges:
        if edge.get("from_node") == "__start__":
            target = edge.get("to_node")
            if isinstance(target, str):
                entry_nodes.add(target)
        if edge.get("condition_key"):
            source = edge.get("from_node")
            if isinstance(source, str) and source != "__start__":
                json_required_nodes.add(source)
    return TopologyInfo(entry_nodes, json_required_nodes)


__all__ = [
    "analyze_topology",
    "auto_complete_entry_exit_edges",
    "normalize_tool_ref",
    "parse_binding",
    "parse_tool_ref",
]
