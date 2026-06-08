"""State Routing for the Graph Compiler.

Provides helpers for node executors to read/write state via configurable keys
instead of hardcoded GraphState fields.

State layout:
    state["dynamic"][key] — all node I/O goes through the dynamic bucket
    state["messages"]     — LLM conversation history (reducer-managed)
    state["final_output"] — graph final output (reducer-managed)

Priority for reads: dynamic[key] → default
"""

from __future__ import annotations

from contextunity.core.narrowing import as_int
from contextunity.core.types import is_object_dict

from contextunity.router.cortex.types import GraphState, StateUpdate

# Top-level keys that bypass the dynamic bucket (reducer-managed fields)
STATE_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {"messages", "final_output", "intermediate_results", "tenant_id", "_last_node"}
)
_TOP_LEVEL_KEYS = STATE_TOP_LEVEL_KEYS


def read_state_input(
    state: GraphState,
    key: str,
    *,
    default: object = None,
) -> object:
    """Read a value from state — dynamic bucket first, top-level fallback.

    Priority: dynamic[key] → state[key] (for reducer-managed keys) → default.

    Args:
        state: LangGraph graph state.
        key: The key to read.
        default: Value to return if key is not found.

    Returns:
        The resolved value, or default.
    """
    raw_dynamic = state.get("dynamic")
    dynamic = raw_dynamic if is_object_dict(raw_dynamic) else {}
    if key in dynamic:
        return dynamic[key]

    # Reducer-managed keys (messages, final_output, etc.) live at the top level.
    if key in _TOP_LEVEL_KEYS:
        val: object | None = state.get(key)
        if val is not None:
            return val

    return default


def read_state_input_mapping(
    state: GraphState,
    mapping: dict[str, str],
) -> StateUpdate:
    """Read multiple keys from state via a mapping.

    Each entry maps: logical_name → state_key.
    E.g. {"query": "user_query", "context": "retrieved_docs"}
    → {"query": state["dynamic"]["user_query"], "context": state["dynamic"]["retrieved_docs"]}

    Missing keys get None.
    """
    result: StateUpdate = {}
    for logical_name, state_key in mapping.items():
        result[logical_name] = read_state_input(state, state_key)
    return result


def write_state_output(
    output_key: str,
    value: object,
    *,
    append: bool = False,
    legacy_keys: frozenset[str] | None = None,
) -> StateUpdate:
    """Build a LangGraph state update dict for the given output key.

    Args:
        output_key: Where to write the value.
        value: The value to write.
        append: If True, wrap value in list for append semantics.
        legacy_keys: Keys that should be written to top-level instead of
                     dynamic. Defaults to _TOP_LEVEL_KEYS.

    Returns:
        State update dict suitable for returning from a LangGraph node.
    """
    keys = legacy_keys if legacy_keys is not None else _TOP_LEVEL_KEYS
    actual_value = [value] if append else value

    # Reducer-managed keys go top-level
    if output_key in keys:
        return {output_key: actual_value}

    # Everything else goes into dynamic bucket
    return {"dynamic": {output_key: actual_value}}


# ── Typed convenience readers ────────────────────────────────────────


def read_state_str(
    state: GraphState,
    key: str,
    *,
    default: str | None = None,
) -> str | None:
    """Read a string value from state, returning default if absent or wrong type."""
    val = read_state_input(state, key)
    if isinstance(val, str):
        return val
    return default


def read_state_int(
    state: GraphState,
    key: str,
    *,
    default: int = 0,
) -> int:
    """Read an int value from state, returning default if absent or wrong type."""
    val = read_state_input(state, key)
    return as_int(val, default=default)


def read_state_dict(
    state: GraphState,
    key: str,
    *,
    default: dict[str, object] | None = None,
) -> dict[str, object]:
    """Read a dict value from state, returning default if absent or wrong type."""
    val = read_state_input(state, key)
    if is_object_dict(val):
        return dict(val)
    return default if default is not None else {}


__all__ = [
    "read_state_dict",
    "read_state_input",
    "read_state_input_mapping",
    "read_state_int",
    "read_state_str",
    "write_state_output",
]
