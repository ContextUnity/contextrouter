"""Memory Injection for the Graph Compiler (Brain Phase C).

Implements `pipeline: memory: true` — auto-injects memory_load and
memory_save platform nodes into compiled graphs.

Injection rewires edges:
    __start__ → first_node  →  __start__ → _memory_load → first_node
    last_node → __end__     →  last_node → _memory_save → __end__

PII scanning is handled internally by the brain_memory_write executor
(via Zero), not as a separate graph node.

Supports retrieval depth tiers (shallow/standard/deep/research)
and experience_lookup configuration for LLM nodes.
"""

from __future__ import annotations

import copy
from typing import ClassVar, Literal

from contextunity.core.exceptions import ConfigurationError
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.cortex.compiler.node_config import NodeConfig
from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    ProjectManifest,
)

# ── Pydantic Config Models ───────────────────────────────────────

# Valid depth tier literals
DepthTier = Literal["shallow", "standard", "deep", "research"]
type RetrievalParamValue = bool | int
type RetrievalParams = dict[str, RetrievalParamValue]


class ExperienceLookupConfig(BaseModel, frozen=True):
    """Bounded parameters controlling Brain experience similarity search on LLM nodes."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    enabled: bool = False
    min_q: float = Field(default=0.7, ge=0.0, le=1.0)
    limit: int = Field(default=3, ge=1, le=100)


class GraphPipelineConfig(BaseModel):
    """Toggle and depth configuration for the ``pipeline:`` manifest section."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    memory: bool = False
    memory_depth: DepthTier = "standard"


# ── Retrieval Depth Tiers ─────────────────────────────────────────

_DEPTH_TIERS: dict[str, RetrievalParams] = {
    "shallow": {
        "include_facts": True,
        "include_episodes": False,
        "include_experiences": False,
        "facts_limit": 10,
    },
    "standard": {
        "include_facts": True,
        "include_episodes": True,
        "include_experiences": False,
        "facts_limit": 20,
        "episodes_limit": 5,
    },
    "deep": {
        "include_facts": True,
        "include_episodes": True,
        "include_experiences": True,
        "facts_limit": 50,
        "episodes_limit": 10,
        "experiences_limit": 10,
        "multi_query_expansion": True,
    },
    "research": {
        "include_facts": True,
        "include_episodes": True,
        "include_experiences": True,
        "facts_limit": 100,
        "episodes_limit": 50,
        "experiences_limit": 25,
        "multi_query_expansion": True,
        "full_history_scan": True,
    },
}


def get_retrieval_params(depth: str | None) -> RetrievalParams:
    """Get retrieval parameters for a given depth tier.

    Args:
        depth: One of 'shallow', 'standard', 'deep', 'research'.
               None defaults to 'standard'.

    Returns:
        Dict of retrieval parameters for brain_memory_read.

    Raises:
        ConfigurationError: If depth is not a valid tier.
    """
    tier = depth or "standard"
    if tier not in _DEPTH_TIERS:
        raise ConfigurationError(
            message=(f"Invalid memory_depth '{tier}'. Valid tiers: {sorted(_DEPTH_TIERS.keys())}"),
        )
    return dict(_DEPTH_TIERS[tier])


# ── Experience Lookup Config ──────────────────────────────────────


def build_experience_lookup_config(config: dict[str, object]) -> dict[str, object]:
    """Build experience lookup parameters from node config.

    Uses ExperienceLookupConfig Pydantic model for validation.
    Raises ValidationError if values are out of bounds.

    Config keys:
        experience_lookup: bool (default: False)
        experience_min_q: float (default: 0.7, range: [0.0, 1.0])
        experience_limit: int (default: 3, range: [1, 100])

    Returns:
        Dict with 'enabled', 'min_q', 'limit'.
    """
    enabled_raw = config.get("experience_lookup", False)
    enabled = enabled_raw if isinstance(enabled_raw, bool) else False
    if not enabled:
        return {"enabled": False}

    min_q_raw = config.get("experience_min_q", 0.7)
    limit_raw = config.get("experience_limit", 3)

    validated = ExperienceLookupConfig(
        enabled=True,
        min_q=min_q_raw if isinstance(min_q_raw, (int, float)) else 0.7,
        limit=limit_raw if isinstance(limit_raw, int) else 3,
    )
    return validated.model_dump()


# ── Memory Node Injection ────────────────────────────────────────


def _make_memory_load_node(graph_config: ProjectManifest) -> CompilerNodeSpec:
    """Build a ``_memory_load`` platform node spec bound to ``brain_memory_read`` with depth-tiered retrieval params."""
    pipeline = graph_config.get("pipeline", {})
    depth = pipeline.get("memory_depth")
    retrieval_params = get_retrieval_params(depth)

    include_facts = retrieval_params.get("include_facts", True)
    include_episodes = retrieval_params.get("include_episodes", True)
    include_experiences = retrieval_params.get("include_experiences", False)
    facts_limit = retrieval_params.get("facts_limit", 20)
    tool_cfg: dict[str, object] = {
        "include_facts": include_facts if isinstance(include_facts, bool) else True,
        "include_episodes": include_episodes if isinstance(include_episodes, bool) else True,
        "include_experiences": (
            include_experiences if isinstance(include_experiences, bool) else False
        ),
        "facts_limit": facts_limit if type(facts_limit) is int else 20,
    }
    if depth:
        tool_cfg["search_depth"] = depth

    mem_cfg = NodeConfig(
        state_output_key="memory",
        tool_config=tool_cfg,
    )

    return {
        "name": "_memory_load",
        "type": "tool",
        "tool_binding": "brain_memory_read",
        "config": mem_cfg.as_manifest_dict(),
    }


def _make_memory_save_node() -> CompilerNodeSpec:
    """Create the _memory_save platform node spec.

    PII scanning is handled internally by the brain_memory_write
    executor via Zero — not exposed as a separate graph node.
    """
    save_cfg = NodeConfig(
        state_input_key="messages",
        tool_config={"pii_scan": True},
    )
    return {
        "name": "_memory_save",
        "type": "tool",
        "tool_binding": "brain_memory_write",
        "config": save_cfg.as_manifest_dict(),
    }


def inject_memory_nodes(
    nodes: list[CompilerNodeSpec],
    edges: list[CompilerEdgeSpec],
    graph_config: ProjectManifest,
) -> tuple[list[CompilerNodeSpec], list[CompilerEdgeSpec]]:
    """Auto-inject memory nodes if pipeline.memory is true.

    Rewires edges:
        __start__ → first_node  →  __start__ → _memory_load → first_node
        last_node → __end__     →  last_node → _memory_save → __end__

    Args:
        nodes: Original node list (not modified in-place).
        edges: Original edge list (not modified in-place).
        graph_config: Graph-level config dict.

    Returns:
        (new_nodes, new_edges) — copies with injected memory nodes.
        If memory not enabled, returns originals unchanged.
    """
    pipeline = graph_config.get("pipeline", {})
    if not pipeline.get("memory"):
        return nodes, edges

    # Deep copy to avoid mutating originals
    new_nodes = copy.deepcopy(nodes)
    new_edges = copy.deepcopy(edges)

    # Create memory nodes
    load_node = _make_memory_load_node(graph_config)
    save_node = _make_memory_save_node()
    new_nodes.append(load_node)
    new_nodes.append(save_node)

    # Find and rewire __start__ → first_node edge
    for edge in new_edges:
        to_val = edge.get("to_node")
        if edge.get("from_node") == "__start__" and to_val:
            edge["to_node"] = "_memory_load"
            new_edges.append({"from_node": "_memory_load", "to_node": to_val})
            break

    # Find and rewire last_node → __end__ edge
    for edge in new_edges:
        from_val = edge.get("from_node")
        if edge.get("to_node") == "__end__" and from_val:
            if from_val == "_memory_load":
                continue
            edge["from_node"] = "_memory_save"
            new_edges.append({"from_node": from_val, "to_node": "_memory_save"})
            break

    return new_nodes, new_edges


__all__ = [
    "inject_memory_nodes",
    "get_retrieval_params",
    "build_experience_lookup_config",
    "ExperienceLookupConfig",
    "GraphPipelineConfig",
    "DepthTier",
]
