"""Sub-Agent Spawner for Router."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Optional

from contextcore import ContextToken, ContextUnit

from contextrouter.modules.grpc import get_worker_client

from .prompt_generator import SubAgentPromptGenerator

logger = logging.getLogger(__name__)


class IsolationContext:
    """Isolation context for sub-agents."""

    def __init__(
        self,
        tenant_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        parent_agent_id: str = "",
        subagent_id: str = "",
    ):
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.trace_id = trace_id or self._generate_trace_id()
        self.parent_agent_id = parent_agent_id
        self.subagent_id = subagent_id

    @staticmethod
    def _generate_trace_id() -> str:
        """Generate a unique trace ID."""
        return uuid.uuid4().hex

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "parent_agent_id": self.parent_agent_id,
            "subagent_id": self.subagent_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IsolationContext:
        """Create from dictionary."""
        return cls(
            tenant_id=data.get("tenant_id"),
            session_id=data.get("session_id"),
            trace_id=data.get("trace_id"),
            parent_agent_id=data.get("parent_agent_id", ""),
            subagent_id=data.get("subagent_id", ""),
        )

    def to_context_unit(self) -> ContextUnit:
        """Convert to ContextUnit for gRPC calls."""
        from contextcore import SecurityScopes

        return ContextUnit(
            payload={},
            provenance=[f"subagent:{self.subagent_id}"],
            security=SecurityScopes(
                read=[f"tenant:{self.tenant_id}:read"] if self.tenant_id else [],
                write=[f"tenant:{self.tenant_id}:write"] if self.tenant_id else [],
            ),
            trace_id=self.trace_id,
        )


class SubAgentSpawner:
    """Spawns sub-agents for task execution."""

    def __init__(self, brain_endpoint: str = "brain.contextunity.ts.net:50051"):
        """Initialize spawner.

        Args:
            brain_endpoint: Brain gRPC endpoint for prompt generation
        """
        self._worker_client = None
        self.prompt_generator = SubAgentPromptGenerator(brain_endpoint=brain_endpoint)

    async def _get_worker_client(self, token: Optional[ContextToken] = None):
        """Get Worker gRPC client.

        Args:
            token: Optional ContextToken for authorization
        """
        if self._worker_client is None or (token and self._worker_client.token != token):
            self._worker_client = await get_worker_client(token=token)
        return self._worker_client

    async def spawn_subagent(
        self,
        parent_agent_id: str,
        task: Dict[str, Any],
        tenant_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        agent_type: str = "task_executor",
        config: Dict[str, Any] | None = None,
        parent_trace_id: str | None = None,  # Trace ID from parent agent
        token: Optional[ContextToken] = None,  # ContextToken for authorization
    ) -> str:
        """Spawn a new sub-agent.

        Args:
            parent_agent_id: ID of parent agent
            task: Task to execute
            tenant_id: Tenant isolation
            session_id: Session isolation
            trace_id: Trace ID for distributed tracing
            agent_type: Type of agent (task_executor, parallel_worker, etc.)
            config: Agent configuration

        Returns:
            Sub-agent ID
        """
        # Generate unique sub-agent ID
        subagent_id = f"{parent_agent_id}:{uuid.uuid4().hex[:8]}"

        # Use parent_trace_id if provided, otherwise use trace_id
        # This ensures sub-agents inherit the parent's trace for distributed tracing
        effective_trace_id = parent_trace_id or trace_id

        # Create isolation context
        isolation_context = IsolationContext(
            tenant_id=tenant_id,
            session_id=session_id,
            trace_id=effective_trace_id,
            parent_agent_id=parent_agent_id,
            subagent_id=subagent_id,
        )

        logger.info("Spawning sub-agent %s for parent %s", subagent_id, parent_agent_id)

        # Generate system prompt/manifest from Brain
        try:
            manifest = await self.prompt_generator.generate_manifest(
                task=task,
                agent_type=agent_type,
                tenant_id=tenant_id or "default",
            )
            # Add manifest to config
            if config is None:
                config = {}
            config["system_prompt"] = manifest
            logger.debug(
                "Generated manifest for sub-agent %s (length=%s)", subagent_id, len(manifest)
            )
        except Exception as e:
            logger.warning("Failed to generate manifest for sub-agent %s: %s", subagent_id, e)
            # Continue without manifest - sub-agent will use default prompt

        # Send spawn request to Worker with token
        await self._send_to_worker(
            subagent_id=subagent_id,
            task=task,
            agent_type=agent_type,
            isolation_context=isolation_context,
            config=config or {},
            token=token,
        )

        return subagent_id

    async def _send_to_worker(
        self,
        subagent_id: str,
        task: Dict[str, Any],
        agent_type: str,
        isolation_context: IsolationContext,
        config: Dict[str, Any],
        token: Optional[ContextToken] = None,
    ) -> None:
        """Send spawn request to Worker.

        Args:
            subagent_id: Sub-agent ID
            task: Task to execute
            agent_type: Type of agent
            isolation_context: Isolation context
            config: Agent configuration
            token: Optional ContextToken for authorization
        """
        try:
            client = await self._get_worker_client(token=token)

            # Create ContextUnit for spawn request
            unit = ContextUnit(
                payload={
                    "subagent_id": subagent_id,
                    "task": task,
                    "agent_type": agent_type,
                    "isolation_context": isolation_context.to_dict(),
                    "config": config,
                },
                provenance=[f"router:spawn:{subagent_id}"],
                trace_id=isolation_context.trace_id,
            )

            # Call Worker StartWorkflow with workflow_type="subagent"
            # Note: This uses StartWorkflow as a generic spawn mechanism
            unit.payload["workflow_type"] = "subagent"
            response = await client.start_workflow(unit)

            logger.info(
                "Spawned sub-agent %s via Worker: %s",
                subagent_id,
                response.payload.get("workflow_id"),
            )

        except Exception as e:
            logger.error("Failed to send spawn request to Worker: %s", e)
            raise
