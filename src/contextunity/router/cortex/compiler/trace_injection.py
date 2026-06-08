"""Trace Node Injection for the Graph Compiler (Phase 4.C).

Auto-injects the ``_trace`` platform-bound tool node before ``__end__`` in
all compiled graphs. Provides execution observability by logging
token usage, timing, tool calls, and security flags to Brain.

Injection rewires edges:
    last_node → __end__  →  last_node → _trace → __end__

Same architectural pattern as ``memory_injection.py``.
PII scanning for trace data handled internally by the executor.

Security by Construction:
- _trace is in RESERVED_NODE_NAMES — cannot be forged by consumers
- Deep copy prevents mutation of original nodes/edges
- Idempotent: re-injection does not duplicate the node
"""

from __future__ import annotations

import copy

from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    ProjectManifest,
)

# ── Trace Node Spec ───────────────────────────────────────────────

_TRACE_NODE_NAME = "_trace"


def _make_trace_node() -> CompilerNodeSpec:
    """Create the _trace platform-bound tool node spec.

    Calls brain_upsert to persist execution telemetry:
    - token_usage: prompt/completion token counts
    - timing_ms: per-node timing data
    - tool_calls: federated tool invocation log
    - security_flags: attenuation violations, PII detections
    """
    return {
        "name": _TRACE_NODE_NAME,
        "type": "tool",
        "tool_binding": "brain_upsert",
        "config": {
            "state_input_key": "trace_data",
            "collection": "execution_traces",
            "include_token_usage": True,
            "include_timing": True,
            "include_tool_calls": True,
            "include_security_flags": True,
        },
    }


def inject_trace_node(
    nodes: list[CompilerNodeSpec],
    edges: list[CompilerEdgeSpec],
    graph_config: ProjectManifest,
) -> tuple[list[CompilerNodeSpec], list[CompilerEdgeSpec]]:
    """Auto-inject _trace node before __end__ in compiled graphs.

    Rewires:
        last_node → __end__  →  last_node → _trace → __end__

    Also handles conditional edges with ``__end__`` targets.

    Args:
        nodes: Original node list (not modified in-place).
        edges: Original edge list (not modified in-place).
        graph_config: Graph-level config dict (reserved for future use).

    Returns:
        (new_nodes, new_edges) — copies with injected trace node.
    """
    _ = graph_config
    # Idempotency: if already injected, return as-is
    if any(n.get("name") == _TRACE_NODE_NAME for n in nodes):
        return nodes, edges

    # Deep copy to avoid mutating originals
    new_nodes = copy.deepcopy(nodes)
    new_edges = copy.deepcopy(edges)

    # Add trace node
    new_nodes.append(_make_trace_node())

    # Rewire direct edges: X → __end__ becomes X → _trace
    # Then add _trace → __end__
    needs_trace_to_end = False

    for edge in new_edges:
        # Direct edge to __end__
        from_node = edge.get("from_node")
        if edge.get("to_node") == "__end__" and from_node:
            if from_node == _TRACE_NODE_NAME:
                continue
            edge["to_node"] = _TRACE_NODE_NAME
            needs_trace_to_end = True

        # Conditional edges targeting __end__
        cond_map = edge.get("condition_map")
        if cond_map:
            for cond_val, target in cond_map.items():
                if target == "__end__":
                    cond_map[cond_val] = _TRACE_NODE_NAME
                    needs_trace_to_end = True

    # Add _trace → __end__ edge (only if we rewired something)
    if needs_trace_to_end:
        new_edges.append({"from_node": _TRACE_NODE_NAME, "to_node": "__end__"})

    return new_nodes, new_edges


__all__ = [
    "inject_trace_node",
]
