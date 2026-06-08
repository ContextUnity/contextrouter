"""Compiler-specific types and schemas.

Defines the structural contracts for the graph compilation pipeline:
- ``NodeConfig`` (Pydantic, see ``node_config.py``) / ``PlatformToolConfig`` — per-node configuration hierarchy
- ``CompilerNodeSpec`` / ``CompilerEdgeSpec`` — graph topology specs
- ``CompilerManifestConfig`` — graph-level manifest settings
- Telemetry event schemas (LLM, Tool, Node)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple, TypeAlias, TypedDict, TypeGuard

from contextunity.core.types import is_json_dict, is_object_dict

# ── Tool Binding Configuration Registry ──────────────────────────
#
# All valid configuration keys for tool bindings (platform + federated),
# grouped by service domain. Each tool executor validates its own subset
# via Pydantic at bind time. Types are widened (Literal → str) since
# runtime validation happens in the executor, not at the TypedDict level.


class ToolBindingConfig(TypedDict, total=False):
    """Registry of all tool binding configuration keys (platform + federated).

    Grouped by service domain. Every key is optional (``total=False``)
    because each tool uses only its own subset.
    """

    # ── Brain ──
    collection: str
    top_k: int
    rerank: bool
    rerank_model: str | None
    similarity_threshold: float
    filter_metadata: dict[str, str]
    memory_scope: str
    last_n: int
    user_id: str | None
    scope_path: str
    created_by: str | None
    ids: list[str]
    entity: str | None
    direction: str
    depth: int
    search_depth: str
    metadata_schema: dict[str, str]
    include_facts: bool
    include_episodes: bool
    include_experiences: bool
    facts_limit: int
    episodes_limit: int
    experiences_limit: int
    multi_query_expansion: bool
    full_history_scan: bool
    pii_scan: bool

    # ── Content Processing ──
    input_key: str
    output_key: str
    taxonomy_key: str
    criteria_key: str
    response_format: Literal["text", "json"]
    confidence_threshold: float
    pass_threshold: float
    threshold: float
    max_candidates: int
    max_items: int
    strict_mode: bool
    language: str

    # ── Search & Retrieval ──
    provider: str
    max_results: int
    search_kwargs: dict[str, str]

    # ── Generation ──
    max_tokens: int
    max_output_tokens: int
    reasoning_effort: str
    system_prompt: str
    custom_tools_key: str

    # ── Formatter ──
    default_format: str
    max_table_rows: int
    include_citations: bool
    include_metadata: bool

    # ── File Download / Ingest ──
    url_key: str
    auth_mode: str
    username_key: str
    password_key: str
    api_key_key: str
    api_key_param: str
    max_size_mb: int
    retries: int

    # ── Language Tool ──
    categories: list[str]
    max_suggestions: int

    # ── Shield ──
    # (uses `categories` from Language Tool — shared key)

    # ── SQL Visualizer ──
    max_chart_points: int

    # ── Worker ──
    workflow_type: str
    workflow_id: str
    task_queue: str
    timeout_seconds: int
    schedule_name: str
    cron_expression: str
    sandbox: bool
    args: dict[str, str]

    # ── Zero (PII) ──
    entity_types: list[str]
    strategy: str
    session_id: str
    mode: str
    preserve_format: bool
    sensitivity_threshold: float


# ── Node Configuration ───────────────────────────────────────────


class NodeMeta(TypedDict, total=False):
    """Node observability metadata passed through to executors."""

    tool_kind: str
    source: str
    toolkit: str


class TemplateNodeOverride(TypedDict, total=False):
    """Per-node keys allowed in ``build_from_template(..., overrides=...)``.

    Mirrors the merge allowlist in ``template_loader`` (``name`` / other unsafe
    keys are rejected at runtime). ``config`` is a partial mapping merged into
    the node's :class:`~contextunity.router.cortex.compiler.node_config.NodeConfig`.
    """

    model: str
    prompt_ref: str
    tool_binding: str
    pii_masking: bool
    response_format: str
    type: Literal["llm", "embeddings", "agent", "tool"]
    config: dict[str, object]


TemplateBuildOverrides: TypeAlias = Mapping[str, TemplateNodeOverride]


_TEMPLATE_OVERRIDE_KEYS = frozenset(
    {
        "model",
        "prompt_ref",
        "tool_binding",
        "pii_masking",
        "response_format",
        "type",
        "config",
    }
)


def is_template_node_override(value: object) -> TypeGuard[TemplateNodeOverride]:
    """Narrow template override blobs to ``TemplateNodeOverride``."""
    if not is_json_dict(value):
        return False
    if not value:
        return False
    for key, item in value.items():
        if key not in _TEMPLATE_OVERRIDE_KEYS:
            return False
        if key == "pii_masking" and not isinstance(item, bool):
            return False
        if key in {"model", "prompt_ref", "tool_binding", "response_format", "type"}:
            if not isinstance(item, str):
                return False
        if key == "config" and not is_json_dict(item):
            return False
    return True


def coerce_manifest_int(value: object) -> int | None:
    """Coerce wire/gRPC numeric values into manifest integer fields."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


# ── Compiler Specs ───────────────────────────────────────────────


class CompilerNodeSpec(TypedDict, total=False):
    """Mutable node specification dictionary consumed by the compilation pipeline.

    ``config`` is a plain ``dict`` at runtime (YAML/JSON, in-place mutation during
    compile). The canonical validated shape is :class:`~contextunity.router.cortex.compiler.node_config.NodeConfig`;
    use :meth:`NodeConfig.model_validate` at execution boundaries — do not type
    this field as ``NodeConfig`` here or static analysis will assume a model instance.
    """

    name: str
    type: Literal["llm", "embeddings", "agent", "tool"]
    mode: str | None
    model: str
    prompt_ref: str
    prompt_variants_ref: str
    prompt_version: str
    prompt_signature: str
    prompt_variants_versions: dict[str, str]
    description: str
    model_secret_ref: str
    goal: str
    persona: str
    config: dict[str, object]
    meta: NodeMeta
    tools: list[str]
    toolkits: list[str]
    tool_name: str
    tool_kind: str
    tool_binding: str
    pii_masking: bool
    allowed_tenants: list[str]
    graph_key: str


class CompilerManifestConfig(TypedDict, total=False):
    """Graph-level defaults section of the manifest — used by the compiler pipeline.

    This is the ``config`` layer in the 3-level resolution hierarchy:
    node config → graph manifest config → router defaults.

    Attributes:
        model: Default LLM model key for all nodes that don't specify their own.
        model_secret_ref: Default Shield lookup path for the LLM API key.
        config: Free-form graph ``config`` map. May mix graph-only keys
            (``{node}_prompt``, ``visualizer_*``, tool-binding maps, …) with
            node-default knobs. Only fields declared on
            :class:`~contextunity.router.cortex.compiler.node_config.NodeConfig`
            are merged into each node by ``config_resolver``; the rest is consumed
            by executors directly from this manifest.
        federated_tool_map: Maps logical tool names used in ``tool_binding``
            to actual registered tool names. Enables manifest portability.
    """

    model: str
    model_secret_ref: str
    goal: str
    persona: str
    config: dict[str, object]
    federated_tool_map: dict[str, str]
    data_sources: list[CompilerDataSourceSpec]
    pipeline: CompilerPipelineConfig


class ServiceDependencyConfig(TypedDict, total=False):
    """Per-service dependency declaration in the project manifest.

    Each service (brain, shield, worker) requires at minimum
    ``enabled: true`` to activate the corresponding platform tools.
    """

    enabled: bool
    url: str
    timeout: int


class CompilerDataSourceSpec(TypedDict, total=False):
    """Data source routing entry available to retrieval and intent nodes."""

    type: Literal["vector", "sql", "federated", "web"]
    binding: str
    description: str
    provider: str
    max_results: int
    search_kwargs: dict[str, object]
    config: dict[str, object]


class CompilerPipelineConfig(TypedDict, total=False):
    """Optional pipeline toggles stored on the top-level manifest."""

    memory: bool
    memory_depth: Literal["shallow", "standard", "deep", "research"]
    reflection: bool
    verification: bool
    visualization: bool
    suggestions: bool


class ProjectManifest(CompilerManifestConfig, total=False):
    """Full project manifest as registered via YAML.

    Extends :class:`CompilerManifestConfig` with the graph topology
    (nodes, edges) and execution constraints. Used by ``config_resolution.py``
    for runtime node lookup and by the builder for graph construction.

    Attributes:
        nodes: List of node specifications in the graph.
        edges: List of edge specifications connecting nodes.
        services: Service dependency declarations (brain, shield, worker).
        max_retries: Maximum retries for cyclic graphs.
        timeout: Graph execution timeout in seconds.
    """

    nodes: list[CompilerNodeSpec]
    edges: list[CompilerEdgeSpec]
    services: dict[str, ServiceDependencyConfig]
    max_retries: int
    timeout: int


def resolve_manifest_max_retries(config: ProjectManifest) -> int | None:
    """Return ``max_retries`` from manifest top level or nested ``config``."""
    top_level = coerce_manifest_int(config.get("max_retries"))
    if top_level is not None:
        return top_level
    nested = config.get("config")
    if is_object_dict(nested):
        return coerce_manifest_int(nested.get("max_retries"))
    return None


class CompilerEdgeSpec(TypedDict, total=False):
    """Edge specification connecting two nodes in a compiled graph.

    YAML manifests use ``from:``/``to:`` — Pydantic normalizes to
    ``from_node``/``to_node`` at the parse boundary (template_loader).
    All internal runtime code uses these Python-safe keys.
    """

    from_node: str
    to_node: str
    condition_key: str
    condition_map: dict[str, str]


# -- Topology analysis result --------------------------------------------------


class TopologyInfo(NamedTuple):
    """Result of graph topology analysis."""

    entry_nodes: set[str]
    """Nodes reachable from __start__."""
    json_required_nodes: set[str]
    """Nodes whose conditional edges require JSON output format."""


# Re-export: single import path ``from ...compiler.types import NodeConfig``.
