"""Payload models for Router gRPC service.
All payloads follow the ContextUnit protocol pattern.
"""

from __future__ import annotations

from typing import Literal

from contextunity.core.manifest.router import RouterEdge, RouterNode
from contextunity.core.sdk.types import StrictPayloadModel
from contextunity.core.types import JsonDict, WireValue
from pydantic import Field, model_validator

from contextunity.router.service.mixins.execution.types import (
    GraphRunConfigInput,
    coerce_graph_run_config_input,
)


class MessagePayload(StrictPayloadModel):
    """Message in conversation."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ExecuteDispatcherPayload(StrictPayloadModel):
    """Payload for ExecuteDispatcher RPC.

    tenant_id is derived from the ContextToken (single point of truth).

    Request payload structure:
    {
        "messages": [{"role": "user", "content": "..."}],
        "session_id": "optional",
        "platform": "api",
        "max_iterations": 10,
        "metadata": {},
        "allowed_tools": ["tool1", "tool2"],  # Optional: restrict tool access
        "denied_tools": ["tool3"]  # Optional: blacklist specific tools
    }
    """

    messages: list[MessagePayload] = Field(..., description="List of messages")
    session_id: str = Field(default="default", description="Session identifier")
    platform: str = Field(default="grpc", description="Platform identifier")
    max_iterations: int = Field(default=10, ge=1, le=50, description="Maximum iterations")
    metadata: JsonDict = Field(default_factory=dict, description="Additional metadata")
    allowed_tools: list[str] | None = Field(
        default=None,
        description=(
            "Optional tool allowlist. Missing derives access from token permissions; "
            "empty list denies all tools; ['*'] allows all visible tools."
        ),
    )
    denied_tools: list[str] = Field(
        default_factory=list,
        description="List of denied tool names (blacklist). Takes precedence over allowed_tools.",
    )
    model_key: str | None = Field(
        default=None,
        description="LLM model key (e.g. 'openai/gpt-5-mini'). Uses Router default if not set.",
    )


class DispatcherResponsePayload(StrictPayloadModel):
    """Payload for dispatcher response.

    Response payload structure:
    {
        "messages": [...],
        "session_id": "...",
        "metadata": {
            "iteration": 1,
            "tools_used": [...],
            "cost": 0.001
        }
    }
    """

    messages: list[JsonDict] = Field(..., description="Response messages")
    session_id: str = Field(..., description="Session identifier")
    metadata: JsonDict = Field(default_factory=dict, description="Execution metadata")


class StreamDispatcherEventPayload(StrictPayloadModel):
    """Payload for streaming dispatcher events.

    Event payload structure:
    {
        "event_type": "agent_start|tool_call|agent_end|error",
        "data": {...},
        "timestamp": "..."
    }
    """

    event_type: str = Field(..., description="Event type")
    data: JsonDict = Field(..., description="Event data")
    timestamp: str | None = Field(None, description="Event timestamp")


class ExecuteAgentPayload(StrictPayloadModel):
    """Payload for ExecuteAgent RPC.

    Execute a specific named graph/agent.
    tenant_id is derived from the ContextToken (single point of truth).

    Request payload structure:
    {
        "agent_id": "nszu_analyst",  # Graph name
        "input": {"messages": [...]},
        "config": {"configurable": {...}}
    }
    """

    agent_id: str = Field(..., description="Agent/Graph identifier")
    input: dict[str, WireValue] = Field(..., description="Input state for the graph")
    config: dict[str, WireValue] = Field(default_factory=dict, description="Runtime configuration")

    @property
    def graph_run_config(self) -> GraphRunConfigInput | None:
        """L4 RunnableConfig subset coerced from :attr:`config` (L3 wire map)."""
        return coerce_graph_run_config_input(self.config)


class ExecuteNodePayload(StrictPayloadModel):
    """Payload for ExecuteNode RPC.

    Execute a single isolated node within a compiled graph.
    tenant_id is derived from ContextToken.

    Request payload structure:
    {
        "graph_name": "retrieval_augmented",
        "node_name": "router_generate",
        "state": {"messages": [...]},
        "config_overrides": {"configurable": {...}}
    }
    """

    graph_name: str = Field(..., description="Compiled graph identifier")
    node_name: str = Field(..., description="Node to execute within the graph")
    state: dict[str, WireValue] = Field(..., description="Input state for the node")
    config_overrides: dict[str, WireValue] = Field(
        default_factory=dict, description="Optional per-call config"
    )

    @property
    def graph_run_config(self) -> GraphRunConfigInput | None:
        """L4 RunnableConfig subset coerced from :attr:`config_overrides` (L3 wire map)."""
        return coerce_graph_run_config_input(self.config_overrides)


class ToolConfig(StrictPayloadModel):
    """Inline tool registration entry in a manifest bundle."""

    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, description="Tool type: sql, bidi, commerce, …")
    description: str = Field(default="", description="Human-readable summary")
    config: dict[str, WireValue] = Field(default_factory=dict, description="Type-specific config")


class GraphEntry(StrictPayloadModel):
    """Graph definition for manifest registration.

    Unified model for graph entries in the multi-graph manifest map.
    Validates that exactly one graph source is defined: inline nodes/edges,
    ``template='yaml:<name>'``, or ``builtin``.

    ``name`` is auto-populated from the graph map key during registration.

    Security: ``extra='forbid'`` prevents payload injection.
    """

    name: str | None = Field(
        None, min_length=1, max_length=128, description="Graph name (auto-set from map key)"
    )
    id: str | None = Field(None, max_length=128, description="Graph ID from manifest")
    template: str | None = Field(
        None,
        min_length=1,
        max_length=128,
        description="YAML template source, e.g. yaml:retrieval_augmented.",
    )
    builtin: Literal["dispatcher"] | None = Field(
        None,
        description="Platform-owned graph source, e.g. dispatcher.",
    )
    overrides: dict[str, dict[str, WireValue]] | None = Field(
        None,
        description="Per-node override map used natively for templates.",
    )
    config_ref: str | None = Field(
        None,
        max_length=256,
        pattern=r"^[a-zA-Z0-9_/.\-]+$",
        description="Reference to external config file",
    )
    config: dict[str, WireValue] = Field(
        default_factory=dict, description="Configuration injected to nodes matching their keys."
    )
    router_callbacks: list[str] | None = Field(
        None, description="Nodes allowed to be called directly via ExecuteNode"
    )
    nodes: list[RouterNode] | None = Field(
        None, description="Declarative graph nodes for inline graph source"
    )
    edges: list[RouterEdge] | None = Field(
        None, description="Declarative graph edges for inline graph source"
    )

    @model_validator(mode="after")
    def validate_template_shape(self) -> "GraphEntry":
        """Enforce mutual exclusion of graph sources (inline / template / builtin).

        Returns:
            Validated ``GraphEntry`` instance.
        """
        from contextunity.core.manifest.helpers import validate_graph_source_shape

        validate_graph_source_shape(
            has_inline=self.nodes is not None or self.edges is not None,
            has_template=self.template is not None,
            has_builtin=self.builtin is not None,
            template=self.template,
            overrides=self.overrides,
            nodes=self.nodes,
            edges=self.edges,
            label="graph entry",
        )
        return self


# Backward-compatible alias — GraphConfig was merged into GraphEntry
GraphConfig = GraphEntry


class RegisterManifestPayload(StrictPayloadModel):
    """Payload for RegisterManifest RPC.

    Expects a pre-compiled bundle from ArtifactGenerator (contextunity.core.manifest).
    Bundle is compiled project-side with secrets resolved from project's os.environ.

    Security: ``extra='forbid'`` prevents payload injection.
    ``bundle`` MUST NOT contain ``project_secret`` — secrets are resolved from env.

    ::

        {
            "bundle": {
                "project_id": "my-project",
                "tenant_id": "my-project",
                "default_graph": "main",
                "graph": {
                    "main": {"template": "yaml:retrieval_augmented"},
                    "gardener": {"template": "yaml:gardener"}
                },
                "tools": [...],
                "policy": {...},
                "secrets": {"openai": "sk-..."}  // present when Shield unavailable
            },
        }
    """

    bundle: JsonDict | None = Field(None, description="Pre-compiled registration bundle")


__all__ = [
    "MessagePayload",
    "ExecuteDispatcherPayload",
    "DispatcherResponsePayload",
    "StreamDispatcherEventPayload",
    "ExecuteAgentPayload",
    "ExecuteNodePayload",
    "ToolConfig",
    "GraphConfig",
    "GraphEntry",
    "RegisterManifestPayload",
]
