"""Cortex-wide shared types.

Types used across the entire cortex pipeline — compiler, dispatcher,
subagents, platform tools, and security nodes. Graph-specific state
definitions live in their respective modules (e.g. ``dispatcher_agent/types.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Hashable, Mapping, Sequence
from typing import (
    Annotated,
    NotRequired,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    runtime_checkable,
)

from contextunity.core import ContextToken
from contextunity.core.types import JsonDict, is_json_dict, is_object_dict, is_object_list
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.message import Messages, add_messages
from typing_extensions import TypedDict

from .compiler.types import CompilerNodeSpec

# -- Reducers ------------------------------------------------------------------


def _merge_dicts(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    """Shallow-merge two dictionaries into a new one.

    This function acts as a LangGraph state reducer for merging partial dictionary
    updates (such as ``dynamic`` or ``intermediate_results``) during node transitions.

    Args:
        a: The existing dictionary.
        b: The new dictionary containing updates.

    Returns:
        A new dictionary containing the shallow-merged keys and values from both inputs,
        with values from ``b`` overriding those in ``a`` in case of overlap.
    """
    return {**a, **b}


GRAPH_MERGE_DICT_KEYS: frozenset[str] = frozenset({"intermediate_results", "dynamic"})


def _merge_token_usage(left: object, right: object) -> dict[str, object]:
    """Reducer for ``_token_usage`` — node updates carry the full accumulated total."""
    if is_object_dict(right) and right:
        return dict(right)
    if is_object_dict(left):
        return dict(left)
    return {}


def _coerce_messages(raw: object) -> list[BaseMessage]:
    if not is_object_list(raw):
        return []
    return [item for item in raw if isinstance(item, BaseMessage)]


def is_graph_state(value: object) -> TypeGuard[GraphState]:
    """Narrow injected/merged runtime dicts to ``GraphState`` at platform boundaries."""
    return is_object_dict(value)


def _is_langgraph_messages(messages: list[BaseMessage]) -> TypeGuard[Messages]:
    """``Messages`` is a LangGraph alias for sequences of message-like objects."""
    return bool(messages) or not messages


def _merge_langgraph_messages(
    left: list[BaseMessage],
    right: list[BaseMessage],
) -> list[BaseMessage]:
    """Apply LangGraph ``add_messages`` and return a typed message list."""
    if not _is_langgraph_messages(left) or not _is_langgraph_messages(right):
        return _coerce_messages(right or left)
    merged_raw: object = add_messages(left, right)
    return _coerce_messages(merged_raw)


def merge_graph_state_update(
    accumulated: dict[str, object],
    update: dict[str, object],
) -> dict[str, object]:
    """Apply a node state update using the same reducers as compiled LangGraph state.

    ``StreamAgent`` reconstructs final graph state from ``astream_events`` node
    outputs. Plain ``dict.update`` drops accumulated ``intermediate_results`` keys
    because each node emits a partial map for that channel.
    """
    merged = dict(accumulated)
    for key, value in update.items():
        if key in GRAPH_MERGE_DICT_KEYS:
            existing = merged.get(key)
            left = existing if is_object_dict(existing) else {}
            right = value if is_object_dict(value) else {}
            merged[key] = _merge_dicts(left, right)
        elif key == "messages":
            left_msgs = _coerce_messages(merged.get("messages"))
            right_msgs = _coerce_messages(value)
            if right_msgs:
                merged[key] = _merge_langgraph_messages(left_msgs, right_msgs)
            elif left_msgs:
                merged[key] = left_msgs
            else:
                merged[key] = []
        else:
            merged[key] = value
    return merged


# -- Base graph state ----------------------------------------------------------


class BaseGraphStateUpdate(TypedDict, total=False):
    """Minimal state update contract shared by all cortex graphs.

    Every graph-specific state update (DispatcherStateUpdate,
    SqlAnalyticsStateUpdate, etc.) should inherit from this so that
    ``make_secure_node`` and shared infrastructure can operate on
    any graph type.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: ExecutionMetadata
    tenant_id: str
    session_id: str
    platform: str
    access_token: ContextToken
    iteration: int
    max_iterations: int
    allowed_tools: list[str]
    denied_tools: list[str]
    trace_id: str | None
    _start_ts: float

    # Token injected by secure_node for platform tool access
    __token__: ContextToken

    # Shared output fields — written by all executor types
    final_output: dict[str, object]
    structured_output: NodeOutput
    intermediate_results: Annotated[NodeResultMap, _merge_dicts]
    _last_node: str
    _raw_output: str
    dynamic: Annotated[dict[str, object], _merge_dicts]
    _token_usage: Annotated[dict[str, object], _merge_token_usage]


class GraphState(TypedDict):
    """Minimal full state contract shared by all cortex graphs.

    Required fields that ``make_secure_node``, telemetry, and tracing
    expect from every graph state.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    metadata: ExecutionMetadata
    tenant_id: str
    session_id: str
    platform: str
    access_token: ContextToken
    iteration: int
    max_iterations: int
    allowed_tools: list[str]
    denied_tools: list[str]
    trace_id: str | None
    _start_ts: float

    # Injected by secure_node — not present at graph construction time
    __token__: NotRequired[ContextToken]

    # Shared output fields — written by all executor types
    final_output: dict[str, object]
    structured_output: NodeOutput
    intermediate_results: Annotated[NodeResultMap, _merge_dicts]
    _last_node: str
    _raw_output: str
    dynamic: Annotated[dict[str, object], _merge_dicts]

    # Accumulated LLM token totals (merged by acc_tokens in node executors).
    _token_usage: NotRequired[Annotated[dict[str, object], _merge_token_usage]]


_StateT = TypeVar("_StateT", bound=GraphState)


class CortexGraph(StateGraph[_StateT]):
    """StateGraph subclass bound to ``GraphState``.

    Inheriting from ``StateGraph[GraphState]`` satisfies ty's
    ``TypedDictLikeV1`` protocol bound structurally, which enables
    clean ``add_node`` calls without any suppression directives.

    Args:
        state_type: The TypedDict schema for this graph.
                    Defaults to ``GraphState``; pass a subclass
                    (e.g. ``DispatcherState``) for domain-specific graphs.
    """

    def __init__(self, state_type: type[_StateT]) -> None:
        """Delegate to ``StateGraph`` using *state_type*."""
        super().__init__(state_type)

    def add_typed_node(self, node: str, action: NodeFunc | Runnable[object, object]) -> None:
        """Add a named node without leaking LangGraph overload details."""
        from .graph_boundary import graph_add_typed_node

        graph_add_typed_node(self, node, action)

    def add_typed_edge(self, start_key: str, end_key: str) -> None:
        """Add a fixed edge between two nodes."""
        _ = super().add_edge(start_key, end_key)

    def add_typed_conditional_edges(
        self,
        source: str,
        path: Callable[[GraphState], Hashable | Sequence[Hashable]],
        path_map: Mapping[str, str],
    ) -> None:
        """Add conditional routing with a simple string path map."""
        from .graph_boundary import graph_add_typed_conditional_edges

        graph_add_typed_conditional_edges(self, source, path, path_map)

    def set_typed_entry_point(self, key: str) -> None:
        """Set the graph entry point using a narrow signature."""
        _ = super().set_entry_point(key)

    def compile_typed(
        self,
        checkpointer: BaseCheckpointSaver[str] | None = None,
    ) -> RunnableGraph:
        """Compile the graph behind a target-local typed wrapper."""
        from .graph_boundary import graph_compile_typed

        return graph_compile_typed(self, checkpointer)


# -- Taxonomy / Ontology JSON schemas ------------------------------------------


class TaxonomyCategoryData(TypedDict, total=False):
    """Single category entry in taxonomy.json."""

    keywords: list[str]


class TaxonomyData(TypedDict, total=False):
    """Schema of taxonomy.json."""

    categories: dict[str, TaxonomyCategoryData]
    canonical_map: dict[str, str]


class OntologyRelations(TypedDict, total=False):
    """Relations section of ontology.json."""

    runtime_fact_labels: list[str]


class OntologyData(TypedDict, total=False):
    """Schema of ontology.json."""

    relations: OntologyRelations


class TokenInfoDict(TypedDict, total=False):
    """Metadata dictionary for tracking token access usage in traces."""

    token_id: str
    tenant_id: str
    owner: str
    user_id: str
    agent_id: str
    user_namespace: str
    permissions: list[str]
    allowed_tenants: list[str]


StateUpdate = dict[str, object]
"""Partial graph state update returned by node executors.

A plain dict because executors write to dynamic keys resolved from
manifest config (e.g. ``state_output_key``). LangGraph merges this
into the full state via its reducer logic.
"""


# -- LangChain boundary protocols ----------------------------------------------


@runtime_checkable
class MessageRoleContent(Protocol):
    """Message object exposing ``role`` and ``content`` attributes."""

    role: str
    content: object


@runtime_checkable
class MessageDumpable(Protocol):
    """Pydantic v2 ``model_dump`` surface for LangChain messages."""

    def model_dump(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        ...


@runtime_checkable
class LegacyMessageDumpable(Protocol):
    """Pydantic v1 ``dict()`` surface for LangChain messages."""

    def dict(self) -> dict[str, object]:
        """Serialize to a plain dict."""
        ...


def extract_message_content(message: object) -> str:
    """Extract plain-text content from LangChain or dict messages."""
    if isinstance(message, BaseMessage):
        message_content: object = getattr(message, "content", "")
        if isinstance(message_content, str):
            return message_content
        return str(message_content)
    if isinstance(message, MessageRoleContent):
        raw = message.content
        return str(raw) if raw is not None else ""
    if is_json_dict(message):
        content_raw: object = message.get("content", "")
        return str(content_raw) if content_raw is not None else ""
    content_obj: object = getattr(message, "content", None)
    if content_obj is not None:
        return str(content_obj)
    return str(message)


def extract_message_role(message: object) -> str | None:
    """Return role string from attribute-style or dict messages."""
    if isinstance(message, MessageRoleContent):
        return message.role
    if is_json_dict(message):
        role_raw: object = message.get("role")
        return str(role_raw) if role_raw is not None else None
    role_obj: object = getattr(message, "role", None)
    if role_obj is None:
        return None
    return str(role_obj)


def serialize_message_object(message: object) -> object:
    """Convert a LangChain message to a JSON-safe dict when possible."""
    if isinstance(message, MessageDumpable):
        return message.model_dump()
    if isinstance(message, LegacyMessageDumpable):
        return message.dict()
    return message


class GraphNodeExecutor(Protocol):
    """Single compiled-graph node runnable (``ExecuteNode`` path)."""

    async def ainvoke(
        self,
        input: object,
        *,
        config: RunnableConfig | None = None,
    ) -> object:
        """Execute this node only."""
        ...


class RunnableGraph(Protocol):
    """Compiled LangGraph boundary used by execution mixins (``ainvoke`` / ``astream_events``)."""

    @property
    def nodes(self) -> dict[str, GraphNodeExecutor]:
        """Registered node executors keyed by manifest node name."""
        ...

    async def ainvoke(
        self,
        input: object,
        *,
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> object:
        """Run the graph to completion."""
        ...

    def astream_events(
        self,
        input: object,
        *,
        config: RunnableConfig | None = None,
        version: str = "v2",
        **kwargs: object,
    ) -> AsyncIterator[dict[str, object]]:
        """Stream LangGraph v2 events."""
        ...


GraphFactoryProduct: TypeAlias = (
    RunnableGraph | StateGraph[GraphState, None, GraphState, GraphState] | object
)
"""Product of ``RunnableGraphFactory.build()`` — compiled runnable, uncompiled ``StateGraph``, or compiled LangGraph instance (narrow with ``is_runnable_graph`` / ``isinstance(..., StateGraph)``)."""


@runtime_checkable
class RunnableGraphFactory(Protocol):
    """Registry graph factory (distinct from Brain ``GraphBuilder`` / ``RouterGraphBuilderError``).

    Registered via ``graph_registry`` and resolved by ``resolve_graph`` (compiled) or
    ``cortex.builder.build_graph`` (uncompiled ``StateGraph``).
    """

    def build(self) -> GraphFactoryProduct:
        """Build or return a graph topology for registry resolution."""
        ...


def is_runnable_graph(value: object) -> TypeGuard[RunnableGraph]:
    """Runtime guard for registry-built compiled graphs (structural, not LangGraph generics)."""
    return (
        callable(getattr(value, "ainvoke", None))
        and callable(getattr(value, "astream_events", None))
        and hasattr(value, "nodes")
    )


class NodeFunc(Protocol):
    """Protocol for graph node executor functions.

    Using a Protocol instead of ``Callable`` avoids positional-only argument
    resolution in Pyright, which perfectly satisfies LangGraph's node bounds.
    """

    async def __call__(self, state: GraphState, config: RunnableConfig) -> StateUpdate:
        """Execute the node logic within the LangGraph pipeline.

        Args:
            state: The current full state of the graph.
            config: LangChain runtime configuration containing callbacks, tags, metadata, etc.

        Returns:
            A partial state update dictionary containing fields to merge into the graph state.
        """
        ...


# -- Dynamic graph output types ------------------------------------------------

NodeOutput = str | dict[str, object] | object
"""Single node output — text, parsed structure, or arbitrary LLM result."""

NodeResultMap = dict[str, NodeOutput]
"""Accumulated results keyed by node name."""


# -- Message types -------------------------------------------------------------


class MessageDict(TypedDict, total=False):
    """Dict-format message input (alternative to BaseMessage)."""

    role: str
    content: str


# -- Registered project configuration (L4 runtime store) -----------------------


class RegisteredToolEntry(TypedDict, total=False):
    """Inline tool registration entry persisted in ``RegisteredProjectConfig.tools``."""

    name: str
    type: str
    description: str
    config: JsonDict


RegisteredGraphMap = dict[str, JsonDict]
"""Serialized graph entries keyed by manifest graph id (values are L2 GraphEntry dicts)."""


class RegisteredProjectConfig(TypedDict, total=False):
    """Runtime projection stored in ``_project_configs[project_id]`` after RegisterManifest.

    This is **not** compile-time :class:`~contextunity.router.cortex.compiler.types.ProjectManifest`.
    Registration copies policy/tools/services/graph from
    :class:`~contextunity.core.manifest.models.RouterRegistrationBundle` and denormalizes
    ``nodes[]`` from all graph entries — read via :func:`~contextunity.router.cortex.config_resolution.get_node_config` (no re-coerce).

    Graph-level goal/persona/model live under ``graph[graph_key]["config"]``, not at the top level.
    """

    policy: JsonDict
    services: JsonDict
    tools: list[RegisteredToolEntry]
    graph: RegisteredGraphMap
    nodes: list[CompilerNodeSpec]
    project_id: str
    allowed_tenants: list[str]


# Backward-compatible alias — prefer RegisteredProjectConfig in new code.
TenantProjectConfig = RegisteredProjectConfig


def is_registered_project_config(value: object) -> TypeGuard[RegisteredProjectConfig]:
    """Narrow runtime L4 config blobs to ``RegisteredProjectConfig``."""
    if not is_object_dict(value):
        return False
    graph_raw = value.get("graph")
    if not is_object_dict(graph_raw):
        return False
    policy_raw = value.get("policy")
    if policy_raw is not None and not is_json_dict(policy_raw):
        return False
    services_raw = value.get("services")
    if services_raw is not None and not is_json_dict(services_raw):
        return False
    tools_raw = value.get("tools")
    if tools_raw is not None:
        if not is_object_list(tools_raw):
            return False
        for item in tools_raw:
            if not is_object_dict(item):
                return False
    nodes_raw = value.get("nodes")
    if nodes_raw is not None:
        if not is_object_list(nodes_raw):
            return False
        for item in nodes_raw:
            if not is_object_dict(item):
                return False
    return True


class ExecutionMetadata(TypedDict, total=False):
    """Metadata passed from client request for execution and observability."""

    langfuse_enabled: bool | str
    langfuse_project_id: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_host: str
    langfuse_trace_id: str
    langfuse_trace_url: str
    agent_id: str
    graph_name: str
    model_key: str
    system_prompt: str
    tenant_id: str
    user_id: str
    session_id: str
    platform: str
    project_id: str
    project_config: RegisteredProjectConfig
    _inner_provenance: list[str | tuple[str, ...]]
    graph_error: str
    allowed_tools: list[str]
    denied_tools: list[str]


JsonMetadata: TypeAlias = JsonDict
"""JSON metadata maps for Brain SDK filters and upsert schemas.

Semantic alias of :data:`~contextunity.core.types.JsonDict` — not a separate
runtime type. Prefer :class:`ExecutionMetadata` for request-scoped execution
fields.
"""


# -- Security types ------------------------------------------------------------


class SecurityFlag(TypedDict, total=False):
    """Event emitted by security_guard_node when a tool call is intercepted."""

    event: str
    tool: str


class BlockedToolCall(TypedDict):
    """A tool call that was denied by the security guard."""

    tool: str
    reason: str
    tool_call_id: str | None


class ConfirmToolCall(TypedDict):
    """A tool call that requires human-in-the-loop confirmation."""

    tool: str
    tool_call_id: str | None
    risk: str
    reason: str


__all__ = [
    "BaseGraphStateUpdate",
    "BlockedToolCall",
    "ConfirmToolCall",
    "CortexGraph",
    "ExecutionMetadata",
    "GraphState",
    "JsonMetadata",
    "LegacyMessageDumpable",
    "MessageDict",
    "MessageDumpable",
    "MessageRoleContent",
    "extract_message_content",
    "extract_message_role",
    "serialize_message_object",
    "NodeFunc",
    "GraphNodeExecutor",
    "is_registered_project_config",
    "is_runnable_graph",
    "GraphFactoryProduct",
    "RunnableGraph",
    "RunnableGraphFactory",
    "NodeOutput",
    "NodeResultMap",
    "SecurityFlag",
    "StateUpdate",
    "RegisteredGraphMap",
    "RegisteredProjectConfig",
    "RegisteredToolEntry",
    "TenantProjectConfig",
]
# ── Telemetry Event Schemas ──────────────────────────────────────

# JSON-serializable telemetry payload — structured data for observability.
TelemetryValue = (
    str | int | float | bool | None | dict[str, "TelemetryValue"] | list["TelemetryValue"]
)


class LLMUsageData(TypedDict, total=False):
    """Token counts, cost estimate, and model identifier attached to ``llm_end`` telemetry events."""

    model: str
    result: str
    input_tokens: int
    output_tokens: int
    total_cost: float
    estimated: bool


class ToolErrorData(TypedDict, total=False):
    """Error details attached to ``tool_error`` telemetry events with retry eligibility flag."""

    message: str
    details: str
    code: str
    retryable: bool


class ToolTelemetryPayload(TypedDict, total=False):
    """Full tool invocation record: status, timing, binding provenance, and result/error payload."""

    status: str
    duration_ms: int
    tool_kind: str
    tool_binding: str
    handler: str
    source: str
    toolkit: str | None
    args: object
    result: object
    error: ToolErrorData | None


class LLMStartEventData(TypedDict, total=False):
    """Model identifier and prompt version sent with ``llm_start`` telemetry events."""

    model: str
    args: str
    prompt_version: str | None


class ToolTelemetryData(TypedDict, total=False):
    """Aggregated tool execution metrics written to trace spans — timing, provenance, and outcome."""

    status: str
    duration_ms: float
    tool_kind: str
    tool_binding: str
    handler: str
    source: str
    toolkit: str
    args: dict[str, str | int | float | bool | None]
    result: TelemetryValue
    error: TelemetryValue


class NodeEventData(TypedDict, total=False):
    """Payload for ``node_start`` / ``node_end`` events identifying the executing node."""

    node: str
    event: str


class TokenEventData(TypedDict, total=False):
    """Payload for incremental ``token`` events carrying streamed LLM output chunks."""

    content: str
