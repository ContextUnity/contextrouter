"""Template Loader for the Graph Compiler (Phase 4).

Loads pre-built graph definitions from YAML template files,
validates schema via Pydantic strict models, and supports
consumer overrides via deep-merge.

Template discovery: ``importlib.resources`` from ``cortex/graphs/compiler/templates/{name}.yaml``.

Security by Construction:
- All models frozen=True, extra='forbid' — no mutation, no injection.
- Literal types on node type, response_format — no arbitrary strings.
- Bounded numerics on timeout, max_retries — no resource exhaustion.
- Override merge blocks node type changes — no privilege escalation.
- Override merge rejects unknown node names — no phantom injection.
"""

from __future__ import annotations

import logging
import re
from io import StringIO
from typing import ClassVar, Literal

from contextunity.core.exceptions import ConfigurationError
from contextunity.core.parsing import yaml_load
from contextunity.core.types import JsonDict, is_json_dict, is_object_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from contextunity.router.cortex.compiler.node_config import (
    NodeConfig,
    merge_node_config_dicts,
)
from contextunity.router.cortex.compiler.types import (
    CompilerEdgeSpec,
    CompilerNodeSpec,
    TemplateBuildOverrides,
)

logger = logging.getLogger(__name__)

# ── Node type enum (typed, not arbitrary string) ──────────────────

NodeType = Literal["llm", "embeddings", "agent", "tool"]
ResponseFormat = Literal["text", "json"]

# Node name: same regex as validation.py — lowercase, alphanumeric, underscores.
_NODE_NAME_RE = re.compile(r"^[_a-z][a-z0-9_]{0,63}$")

# Safe override keys — only these can be modified via consumer overrides.
# Prevents injection of 'name' (identity) or 'type' (privilege) via merge.
_SAFE_OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "prompt_ref",
        "tool_binding",
        "pii_masking",
        "response_format",
        "config",
    }
)


# ── Pydantic Models (Strict Hardening) ────────────────────────────


class _StrictFrozenTemplateModel(BaseModel):
    """Shared strict base model for template schema objects."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class TemplateNode(_StrictFrozenTemplateModel):
    """Single node in a graph template.

    Invariants:
    - type is restricted to known executor types via Literal
    - name validated against ^[_a-z][a-z0-9_]{0,63}$
    - response_format restricted to text/json
    - config validated via :class:`NodeConfig` (``extra="forbid"``; provider
      knobs under ``provider_config``)
    - tool_binding required for tool nodes (validated downstream)
    """

    name: str
    type: NodeType = "llm"
    model: str | None = None
    prompt_ref: str | None = None
    tool_binding: str | None = None
    pii_masking: bool = False
    response_format: ResponseFormat | None = None
    config: NodeConfig = Field(default_factory=NodeConfig)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not _NODE_NAME_RE.match(v):
            msg = f"Invalid node name '{v}'. Must match ^[_a-z][a-z0-9_]{{{{0,63}}}}$"
            raise ValueError(msg)
        return v

    def to_spec(self) -> CompilerNodeSpec:
        """Convert to CompilerNodeSpec TypedDict for the compilation pipeline."""
        spec: CompilerNodeSpec = {"name": self.name, "type": self.type}
        if self.model is not None:
            spec["model"] = self.model
        if self.prompt_ref is not None:
            spec["prompt_ref"] = self.prompt_ref
        if self.tool_binding is not None:
            spec["tool_binding"] = self.tool_binding
        cfg = self.config.as_manifest_dict()
        if cfg:
            spec["config"] = cfg
        return spec


class TemplateEdge(_StrictFrozenTemplateModel):
    """Single edge in a graph template.

    Supports both direct edges (from→to) and conditional edges
    (from→condition_key→condition_map).
    """

    from_node: str
    to_node: str | None = None
    condition_key: str | None = None
    condition_map: dict[str, str] | None = None

    def to_spec(self) -> CompilerEdgeSpec:
        """Convert to CompilerEdgeSpec TypedDict for the compilation pipeline."""
        spec: CompilerEdgeSpec = {"from_node": self.from_node}
        if self.to_node is not None:
            spec["to_node"] = self.to_node
        if self.condition_key is not None:
            spec["condition_key"] = self.condition_key
        if self.condition_map is not None:
            spec["condition_map"] = self.condition_map
        return spec


class TemplateDefaults(_StrictFrozenTemplateModel):
    """Default LLM provider settings merged into every node unless overridden."""

    model: str | None = None
    model_secret_ref: str | None = None
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)

    @field_validator("model_secret_ref")
    @classmethod
    def _validate_secret_ref(cls, v: str | None) -> str | None:
        if v is not None and not re.match(r"^[a-zA-Z0-9_-]+$", v):
            msg = (
                f"Invalid model_secret_ref '{v}'. "
                "Must be alphanumeric with underscores/hyphens only — "
                "path separators forbidden (traversal prevention)."
            )
            raise ValueError(msg)
        return v


class DataSourceDefinition(_StrictFrozenTemplateModel):
    """Configuration for an integrated data source.

    Each data source maps an intent route to a platform tool binding.
    The intent_analyzer selects sources based on query analysis.
    """

    type: Literal["vector", "sql", "federated", "web"]
    binding: str = Field(..., min_length=1, max_length=128)
    description: str = Field("", max_length=512)
    config: JsonDict = Field(default_factory=dict)


class PipelineToggles(_StrictFrozenTemplateModel):
    """Feature flags for optional pipeline stages.

    Projects enable/disable stages via manifest config.
    Disabled stages are excluded from the compiled graph.
    """

    memory: bool = Field(default=False, description="Auto-inject memory_load/memory_save nodes")
    reflection: bool = Field(default=True, description="Enable reflect node for response quality")
    verification: bool = Field(default=False, description="Enable SQL verify step")
    visualization: bool = Field(default=False, description="Enable SQL visualize step")
    suggestions: bool = Field(default=True, description="Enable follow-up suggestions")


class TemplateConfig(_StrictFrozenTemplateModel):
    """Bounded execution limits, data source routes, and pipeline feature flags for a template."""

    max_retries: int = Field(default=0, ge=0, le=10)
    timeout: int = Field(default=60, ge=1, le=600)
    data_sources: list[DataSourceDefinition] = Field(default_factory=list)
    pipeline: PipelineToggles = Field(default_factory=PipelineToggles)


class TemplateDefinition(_StrictFrozenTemplateModel):
    """Complete graph template definition.

    Loaded from YAML, immutable after construction.
    Schema enforced at load time — malformed templates rejected.
    """

    name: str
    version: str
    description: str = ""
    nodes: list[TemplateNode]
    edges: list[TemplateEdge]
    defaults: TemplateDefaults = Field(default_factory=TemplateDefaults)
    config: TemplateConfig = Field(default_factory=TemplateConfig)


# ── Template Loading ──────────────────────────────────────────────


def _parse_yaml(raw: str, template_name: str) -> dict[str, object]:
    """Parse YAML string, fail-closed on parse errors."""
    try:
        data = yaml_load(StringIO(raw))
    except Exception as exc:
        raise ConfigurationError(
            message=f"Template '{template_name}' contains invalid YAML",
        ) from exc

    if not is_json_dict(data):
        raise ConfigurationError(
            message=(
                f"Template '{template_name}' must be a JSON-compatible YAML mapping, "
                f"got {type(data).__name__}"
            ),
        )
    return {str(key): value for key, value in data.items()}


def _normalize_edges(raw_edges: list[dict[str, object]]) -> list[dict[str, object]]:
    """Normalize edge format: 'from' → 'from_node', 'to' → 'to_node'."""
    normalized: list[dict[str, object]] = []
    for edge in raw_edges:
        norm = dict(edge)
        if "from" in norm:
            norm["from_node"] = norm.pop("from")
        if "to" in norm:
            norm["to_node"] = norm.pop("to")
        normalized.append(norm)
    return normalized


def load_template(template_name: str) -> TemplateDefinition:
    """Load and validate a graph template by name.

    Discovery via ``importlib.resources`` from the ``cortex/graphs/compiler/templates`` package.

    Args:
        template_name: Template identifier (e.g., 'retrieval_augmented', 'gardener').

    Returns:
        Validated, frozen TemplateDefinition.

    Raises:
        ConfigurationError: Template not found, invalid YAML, or schema violation.
    """
    # Locate template resource
    templates_pkg = "contextunity.router.cortex.compiler.templates"
    filename = f"{template_name}.yaml"

    try:
        from importlib.resources import files

        ref = files(templates_pkg).joinpath(filename)
        raw = ref.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError, TypeError) as exc:
        raise ConfigurationError(
            message=f"Template '{template_name}' not found in {templates_pkg}",
        ) from exc

    # Parse YAML
    data = _parse_yaml(raw, template_name)

    # Normalize edge keys (from/to → from_node/to_node for Pydantic)
    raw_edges = data.get("edges")
    if raw_edges is not None:
        if not is_object_list(raw_edges):
            raise ConfigurationError(
                message=f"Template '{template_name}' field 'edges' must be a list.",
            )
        normalized_input: list[dict[str, object]] = []
        for edge in raw_edges:
            if not is_object_dict(edge):
                raise ConfigurationError(
                    message=f"Template '{template_name}' contains a non-mapping edge entry.",
                )
            normalized_input.append({str(key): value for key, value in edge.items()})
        data["edges"] = _normalize_edges(normalized_input)

    # Validate via Pydantic (fail-closed)
    try:
        template = TemplateDefinition.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(
            message=f"Template '{template_name}' schema validation failed",
        ) from exc

    logger.info(
        "📋 Loaded template '%s' v%s (%d nodes, %d edges)",
        template.name,
        template.version,
        len(template.nodes),
        len(template.edges),
    )
    return template


# ── Override Merge ────────────────────────────────────────────────


def merge_overrides(
    template: TemplateDefinition,
    overrides: TemplateBuildOverrides,
) -> TemplateDefinition:
    """Deep-merge consumer overrides onto a template.

    Security invariants:
    - Unknown node names → ConfigurationError (no phantom injection)
    - Node type changes → ConfigurationError (no privilege escalation)
    - Config is deep-merged (override keys replace, missing preserved)
    - Result is a new frozen TemplateDefinition

    Args:
        template: Base template definition.
        overrides: Per-node override dict. Keys = node names.

    Returns:
        New TemplateDefinition with merged values.

    Raises:
        ConfigurationError: Unknown node name or type change attempt.
    """
    if not overrides:
        return template

    # Validate: all override keys must reference existing nodes
    node_names = {n.name for n in template.nodes}
    unknown = set(overrides.keys()) - node_names
    if unknown:
        raise ConfigurationError(
            message=(
                f"Override references non-existent nodes: {sorted(unknown)}. "
                f"Template '{template.name}' has nodes: {sorted(node_names)}"
            ),
        )

    # Build merged node list
    merged_nodes: list[TemplateNode] = []
    for node in template.nodes:
        if node.name not in overrides:
            merged_nodes.append(node)
            continue

        node_overrides = overrides[node.name]

        # Security: block type escalation
        if "type" in node_overrides and node_overrides["type"] != node.type:
            raise ConfigurationError(
                message=(
                    f"Cannot change node type via override: "
                    f"'{node.name}' is '{node.type}', "
                    f"override requests '{node_overrides['type']}'. "
                    f"Type changes can enable privilege escalation."
                ),
            )

        # Security: reject unsafe override keys (prevents name/type injection)
        unsafe_keys = set(node_overrides.keys()) - _SAFE_OVERRIDE_KEYS - {"type"}
        if unsafe_keys:
            raise ConfigurationError(
                message=(
                    f"Override for node '{node.name}' contains unsafe keys: "
                    f"{sorted(unsafe_keys)}. "
                    f"Allowed override keys: {sorted(_SAFE_OVERRIDE_KEYS)}"
                ),
            )

        override_config = node_overrides.get("config")
        merged_config = node.config
        if override_config is not None:
            merged = merge_node_config_dicts(
                node.config.as_manifest_dict(),
                override_config,
            )
            merged_config = NodeConfig.model_validate(merged)

        override_response_format = node_overrides.get("response_format")
        response_format = node.response_format
        if override_response_format in ("text", "json"):
            response_format = override_response_format

        merged_nodes.append(
            TemplateNode(
                name=node.name,
                type=node.type,
                model=node_overrides.get("model", node.model),
                prompt_ref=node_overrides.get("prompt_ref", node.prompt_ref),
                tool_binding=node_overrides.get("tool_binding", node.tool_binding),
                pii_masking=node_overrides.get("pii_masking", node.pii_masking),
                response_format=response_format,
                config=merged_config,
            )
        )

    # Reconstruct template with merged nodes
    return TemplateDefinition(
        name=template.name,
        version=template.version,
        description=template.description,
        nodes=merged_nodes,
        edges=list(template.edges),
        defaults=template.defaults,
        config=template.config,
    )


__all__ = [
    "TemplateDefinition",
    "TemplateNode",
    "TemplateEdge",
    "TemplateDefaults",
    "TemplateConfig",
    "DataSourceDefinition",
    "PipelineToggles",
    "NodeType",
    "ResponseFormat",
    "load_template",
    "merge_overrides",
]
