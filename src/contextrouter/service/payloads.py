"""Payload models for Router gRPC service.

All payloads follow the ContextUnit protocol pattern.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    """Message in conversation."""

    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ExecuteDispatcherPayload(BaseModel):
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
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="List of allowed tool names. Empty list = all tools allowed. ['*'] = all tools.",
    )
    denied_tools: list[str] = Field(
        default_factory=list,
        description="List of denied tool names (blacklist). Takes precedence over allowed_tools.",
    )
    model_key: str | None = Field(
        default=None,
        description="LLM model key (e.g. 'openai/gpt-5-mini'). Uses Router default if not set.",
    )


class DispatcherResponsePayload(BaseModel):
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

    messages: list[dict[str, Any]] = Field(..., description="Response messages")
    session_id: str = Field(..., description="Session identifier")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class StreamDispatcherEventPayload(BaseModel):
    """Payload for streaming dispatcher events.

    Event payload structure:
    {
        "event_type": "agent_start|tool_call|agent_end|error",
        "data": {...},
        "timestamp": "..."
    }
    """

    event_type: str = Field(..., description="Event type")
    data: dict[str, Any] = Field(..., description="Event data")
    timestamp: str | None = Field(None, description="Event timestamp")


class ExecuteAgentPayload(BaseModel):
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
    input: dict[str, Any] = Field(..., description="Input state for the graph")
    config: dict[str, Any] = Field(default_factory=dict, description="Runtime configuration")


class ToolConfig(BaseModel):
    """Configuration for a project tool to register in Router.

    The ``config`` dict carries type-specific settings.
    For ``type="sql"``, expected keys:
        database_url, schema_description, read_only, max_rows, statement_timeout_ms
    """

    name: str = Field(..., min_length=1, description="Unique tool name")
    type: str = Field(..., description="Tool type: sql, search, custom")
    description: str = Field(default="", description="Human-readable tool description")
    config: dict[str, Any] = Field(default_factory=dict, description="Type-specific config")


class GraphConfig(BaseModel):
    """Graph configuration for a project.

    v1alpha runtime supports template graphs only: provide ``template`` + ``config``
    to use a built-in graph builder in Router.
    """

    name: str = Field(..., min_length=1, description="Graph name (used as registry key)")
    template: str = Field(..., description="Built-in template name OR 'custom' for dynamic graph.")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Template config (prompts, tool_bindings, etc.)"
    )
    nodes: list[dict[str, Any]] | None = Field(None, description="Declarative graph nodes")
    edges: list[dict[str, Any]] | None = Field(None, description="Declarative graph edges")


class RegisterManifestPayload(BaseModel):
    """Payload for RegisterManifest RPC.

    Expects a pre-compiled bundle from ArtifactGenerator (contextcore.manifest).
    Bundle is compiled project-side with secrets resolved from project's os.environ.

    {
        "bundle": {
            "project_id": "my-project",
            "tenant_id": "my-project",
            "graph": {...},
            "tools": [...],
            "policy": {...},
            "secrets": {"openai": "sk-..."}  // present when Shield unavailable
        },
        "hash": "a1b2c3d4..."
    }
    """

    bundle: dict[str, Any] | None = Field(None, description="Pre-compiled registration bundle")
    hash: str | None = Field(None, description="Optional hash for idempotent validation")


__all__ = [
    "MessagePayload",
    "ExecuteDispatcherPayload",
    "DispatcherResponsePayload",
    "StreamDispatcherEventPayload",
    "ToolConfig",
    "GraphConfig",
    "RegisterManifestPayload",
]
