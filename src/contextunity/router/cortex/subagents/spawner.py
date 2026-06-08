"""Sub-agent spawner -- instantiates child agent graphs with scoped tokens and tool subsets."""

from __future__ import annotations

import uuid

from contextunity.core import ContextToken, WorkerClient, get_contextunit_logger

from contextunity.router.modules.grpc import get_worker_client

from .prompt_generator import SubAgentPromptGenerator
from .types import IsolationContext, SpawnTask, SubAgentConfig

logger = get_contextunit_logger(__name__)


class SubAgentSpawner:
    """Instantiates child agent graphs with scoped tokens and tool subsets.

    Generates system prompts via Brain semantic/procedural memory, then
    delegates execution to the Worker service via gRPC ``StartWorkflow``.
    """

    _worker_client: WorkerClient | None
    _worker_token: ContextToken | None
    prompt_generator: SubAgentPromptGenerator

    def __init__(self, brain_endpoint: str = "brain.contextunity.ts.net:50051"):
        """Initialize spawner.

        Args:
            brain_endpoint: Brain gRPC endpoint for prompt generation
        """
        self._worker_client = None
        self._worker_token = None
        self.prompt_generator = SubAgentPromptGenerator(brain_endpoint=brain_endpoint)

    async def _get_worker_client(self, token: ContextToken | None = None) -> WorkerClient:
        """Get Worker gRPC client.

        Args:
            token: ContextToken for authorization
        """
        if token is not None:
            if self._worker_client is None or self._worker_token != token:
                self._worker_client = await get_worker_client(token=token)
                self._worker_token = token
            return self._worker_client

        if self._worker_client is None or self._worker_token is not None:
            self._worker_client = await get_worker_client()
            self._worker_token = None
        return self._worker_client

    async def spawn_subagent(
        self,
        parent_agent_id: str,
        task: SpawnTask,
        tenant_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
        agent_type: str = "task_executor",
        config: SubAgentConfig | None = None,
        parent_trace_id: str | None = None,
        token: ContextToken | None = None,
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
        task: SpawnTask,
        agent_type: str,
        isolation_context: IsolationContext,
        config: SubAgentConfig,
        token: ContextToken | None = None,
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
            if token is None:
                from contextunity.router.core.context import get_current_access_token

                token = get_current_access_token()

            client = await self._get_worker_client(token=token)

            response = await client.start_workflow(
                workflow_type="subagent",
                wire={
                    "subagent_id": subagent_id,
                    "task": task,
                    "agent_type": agent_type,
                    "isolation_context": isolation_context.to_dict(),
                    "config": config,
                },
                provenance=[f"router:spawn:{subagent_id}"],
                trace_id=uuid.UUID(isolation_context.trace_id),
            )

            logger.info(
                "Spawned sub-agent %s via Worker: %s",
                subagent_id,
                response.get("workflow_id"),
            )

        except Exception as e:
            logger.error("Failed to send spawn request to Worker: %s", e)
            raise
