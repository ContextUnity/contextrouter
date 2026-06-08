"""Named type aliases for execution mixins.

Centralizes domain types used across execution helpers, agent/dispatcher/node
mixins to avoid anonymous ``dict[str, ...]`` proliferation.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

from contextunity.core.types import (
    JsonDict,
    WireValue,
    is_json_dict,
    is_object_list,
)
from langchain_core.callbacks.base import BaseCallbackHandler
from typing_extensions import TypedDict

from contextunity.router.cortex.types import (
    ExecutionMetadata,
    RegisteredProjectConfig,
    RunnableGraph,
    StateUpdate,
)
from contextunity.router.modules.observability import LangfuseRequestCtx
from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.service.shield_check import ShieldCheckResult

# ---------------------------------------------------------------------------
# Semantic aliases — give domain meaning to generic structures
# ---------------------------------------------------------------------------

#: Per-project registration config built during RegisterManifest / Redis restore.
TenantConfig = RegisteredProjectConfig

#: Map of project_id → registration config (``RouterRegistrationBundle.project_id``).
ProjectConfigMap = dict[str, RegisteredProjectConfig]

#: Map of project_id → {graph_key → registry_name}.
#: Tracks which named graphs each project registered.
ProjectGraphMap = dict[str, dict[str, str]]

#: Map of project_id → [tool_name, ...].
#: Tracks which tools each project registered.
ProjectToolMap = dict[str, list[str]]

#: Map of project_id → {graph_key → [callback_name, ...]}.
#: Tracks router_callbacks declared in each graph entry during RegisterManifest.
RouterCallbackMap = dict[str, dict[str, list[str]]]

#: Graph input dict (messages, metadata, user-provided keys).
GraphInput = dict[str, object]


#: Final graph execution result state (messages, _token_usage, etc.).
GraphResult: TypeAlias = StateUpdate

MAX_CLIENT_MAX_CONCURRENCY = 64
"""Upper bound for caller-supplied LangGraph concurrency to avoid request-level DoS."""

MAX_CLIENT_RECURSION_LIMIT = 100
"""Upper bound for caller-supplied recursion_limit when no manifest budget is stricter."""


#: Client-supplied LangChain ``RunnableConfig`` subset on ExecuteAgent / ExecuteNode RPCs.
#: Coerce from L3 ``dict[str, WireValue]`` via :func:`coerce_graph_run_config_input` only.
class GraphRunConfigInput(TypedDict, total=False):
    """RunnableConfig keys accepted from gRPC execution payloads."""

    tags: list[str]
    run_name: str
    max_concurrency: int
    recursion_limit: int
    metadata: JsonDict
    configurable: JsonDict


def coerce_graph_run_config_input(
    raw: dict[str, WireValue] | GraphRunConfigInput | None,
) -> GraphRunConfigInput | None:
    """Coerce L3 wire config to L4 ``GraphRunConfigInput`` (single choke point)."""
    if not raw:
        return None
    out: GraphRunConfigInput = {}
    tags_raw = raw.get("tags")
    if is_object_list(tags_raw):
        tags: list[str] = []
        for item in tags_raw:
            if item is not None:
                tags.append(str(item))
        if tags:
            out["tags"] = tags
    run_name = raw.get("run_name")
    if isinstance(run_name, str):
        out["run_name"] = run_name
    max_concurrency = raw.get("max_concurrency")
    if (
        isinstance(max_concurrency, int)
        and not isinstance(max_concurrency, bool)
        and 1 <= max_concurrency <= MAX_CLIENT_MAX_CONCURRENCY
    ):
        out["max_concurrency"] = max_concurrency
    recursion_limit = raw.get("recursion_limit")
    if (
        isinstance(recursion_limit, int)
        and not isinstance(recursion_limit, bool)
        and 1 <= recursion_limit <= MAX_CLIENT_RECURSION_LIMIT
    ):
        out["recursion_limit"] = recursion_limit
    metadata = raw.get("metadata")
    if is_json_dict(metadata):
        out["metadata"] = dict(metadata)
    configurable = raw.get("configurable")
    if is_json_dict(configurable):
        out["configurable"] = dict(configurable)
    return out or None


#: Security context flags for trace persistence.
#: Rendered as badges in contextunity.view dashboard.
#: Keys: shield_enabled, shield_mode, pii_masking_enabled, redis_enabled, redis_tls,
#:        redis_encrypted, error.
SecurityFlags = dict[str, bool | str]


# ---------------------------------------------------------------------------
# Structured return type
# ---------------------------------------------------------------------------


class ExecutionContext(NamedTuple):
    """Prepared execution context returned by :func:`prepare_execution`."""

    execution_input: GraphInput
    metadata: ExecutionMetadata
    effective_user_id: str | None
    callbacks: list[BaseCallbackHandler]
    auto_tracer: BrainAutoTracer
    langfuse_ctx: LangfuseRequestCtx


class ResolvedGraph(NamedTuple):
    """Result of :func:`~contextunity.router.service.mixins.execution.helpers.resolve_graph`."""

    name: str
    graph: RunnableGraph


__all__ = [
    "ExecutionContext",
    "ExecutionMetadata",
    "GraphInput",
    "GraphResult",
    "GraphRunConfigInput",
    "ProjectConfigMap",
    "ProjectGraphMap",
    "ProjectToolMap",
    "ResolvedGraph",
    "RouterCallbackMap",
    "RunnableGraph",
    "SecurityFlags",
    "ShieldCheckResult",
    "TenantConfig",
    "coerce_graph_run_config_input",
]
