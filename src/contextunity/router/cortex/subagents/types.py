"""Type definitions for sub-agents -- state schemas, result containers, and orchestration contracts."""

from __future__ import annotations

import uuid
from typing import TypedDict

from contextunity.core.types import JsonDict


class SpawnTask(TypedDict, total=False):
    """Task descriptor passed to the sub-agent spawner.

    Describes the work unit and optional context payload that a child agent
    should execute.
    """

    description: str
    context: str | dict[str, object]


class KnowledgeChunk(TypedDict):
    """A single chunk of knowledge retrieved from Brain semantic or procedural memory.

    Attributes:
        content: The text content of the knowledge fragment.
        metadata: Source metadata (collection, document ID, ingestion timestamp, etc.).
        score: Similarity score from the vector search, higher is more relevant.
    """

    content: str
    metadata: JsonDict
    score: float


class SubAgentConfig(TypedDict, total=False):
    """Configuration dictionary controlling sub-agent behavior.

    Attributes:
        system_prompt: The generated or overridden system prompt (manifest).
        max_iterations: Maximum iterations the sub-agent may execute before termination.
        allowed_tools: Explicit allowlist of tool names the sub-agent may invoke.
        denied_tools: Denylist of tool names the sub-agent must not invoke.
        strategy: Orchestration strategy hint (parallel, sequential, map_reduce).
    """

    system_prompt: str
    max_iterations: int
    allowed_tools: list[str]
    denied_tools: list[str]
    strategy: str


class SubAgentResult(TypedDict, total=False):
    """Result container from a single sub-agent execution.

    Attributes:
        subagent_id: Unique identifier of the sub-agent instance.
        status: Terminal status ("completed", "failed", "timeout").
        result: The output payload produced by the sub-agent.
    """

    subagent_id: str
    status: str
    result: dict[str, object]


class OrchestrationResult(TypedDict, total=False):
    """Aggregated results from an orchestrated batch of sub-agent executions.

    Attributes:
        total: Total number of sub-agents that were spawned.
        results: All sub-agent results regardless of status.
        successful: Sub-agents that completed successfully.
        failed: Sub-agents that terminated with an error.
    """

    total: int
    results: list[SubAgentResult]
    successful: list[SubAgentResult]
    failed: list[SubAgentResult]


class IsolationContext:
    """Multi-tenant isolation context propagated to child sub-agents.

    Carries tenant, session, and trace identifiers to ensure sub-agent
    operations remain scoped to the correct security and observability boundaries.
    """

    def __init__(
        self,
        tenant_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_agent_id: str = "",
        subagent_id: str = "",
    ):
        """Initialize a new IsolationContext.

        Args:
            tenant_id: Tenant identifier for data isolation.
            session_id: Session identifier for conversation continuity.
            trace_id: Distributed trace identifier. Auto-generated if not provided.
            parent_agent_id: Identifier of the parent agent that spawned this sub-agent.
            subagent_id: Unique identifier assigned to the child sub-agent.
        """
        self.tenant_id: str | None = tenant_id
        self.session_id: str | None = session_id
        self.trace_id: str = trace_id or self._generate_trace_id()
        self.parent_agent_id: str = parent_agent_id
        self.subagent_id: str = subagent_id

    @staticmethod
    def _generate_trace_id() -> str:
        """Generate a unique trace ID as a 32-character hex string.

        Returns:
            A UUID4-based hex string suitable for distributed tracing.
        """
        return uuid.uuid4().hex

    def to_dict(self) -> dict[str, str | None]:
        """Serialize the isolation context to a plain dictionary.

        Returns:
            A dictionary with all context fields, suitable for JSON serialization.
        """
        return {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "parent_agent_id": self.parent_agent_id,
            "subagent_id": self.subagent_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | None]) -> IsolationContext:
        """Reconstruct an IsolationContext from a serialized dictionary.

        Args:
            data: A dictionary previously produced by ``to_dict()``.

        Returns:
            A new IsolationContext instance populated from the dictionary values.
        """
        return cls(
            tenant_id=data.get("tenant_id"),
            session_id=data.get("session_id"),
            trace_id=data.get("trace_id"),
            parent_agent_id=data.get("parent_agent_id") or "",
            subagent_id=data.get("subagent_id") or "",
        )
