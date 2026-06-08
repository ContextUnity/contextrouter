"""Sub-agent orchestrator -- coordinates parallel child agent execution with result aggregation."""

from __future__ import annotations

from collections.abc import Coroutine

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError

from .spawner import SubAgentSpawner
from .types import OrchestrationResult, SpawnTask, SubAgentResult

logger = get_contextunit_logger(__name__)


class SubAgentOrchestrator:
    """Coordinates multi-strategy sub-agent execution with result aggregation.

    Supports parallel, sequential, and map-reduce orchestration patterns.
    Each strategy spawns sub-agents via SubAgentSpawner, waits for completion,
    and returns aggregated results.
    """

    spawner: SubAgentSpawner

    def __init__(self):
        """Initialize the orchestrator with a default SubAgentSpawner."""
        self.spawner = SubAgentSpawner()

    async def orchestrate_subagents(
        self,
        parent_agent_id: str,
        tasks: list[SpawnTask],
        strategy: str = "parallel",
        tenant_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> OrchestrationResult:
        """Orchestrate multiple sub-agents.

        Args:
            parent_agent_id: Parent agent ID
            tasks: List of tasks
            strategy: Orchestration strategy
            tenant_id: Tenant isolation
            session_id: Session isolation
            trace_id: Trace ID for distributed tracing

        Returns:
            Aggregated results
        """
        logger.info("Orchestrating %s sub-agents with strategy: %s", len(tasks), strategy)

        if strategy == "parallel":
            return await self._orchestrate_parallel(
                parent_agent_id=parent_agent_id,
                tasks=tasks,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )

        if strategy == "sequential":
            return await self._orchestrate_sequential(
                parent_agent_id=parent_agent_id,
                tasks=tasks,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )

        if strategy == "map_reduce":
            return await self._orchestrate_map_reduce(
                parent_agent_id=parent_agent_id,
                tasks=tasks,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )

        raise ConfigurationError(f"Unknown sub-agent orchestration strategy: {strategy}")

    async def _orchestrate_parallel(
        self,
        parent_agent_id: str,
        tasks: list[SpawnTask],
        tenant_id: str | None,
        session_id: str | None,
        trace_id: str | None,
    ) -> OrchestrationResult:
        """Spawn all sub-agents concurrently and wait for all to complete.

        Args:
            parent_agent_id: Identifier of the parent agent.
            tasks: List of task descriptors to distribute.
            tenant_id: Tenant identifier for data isolation.
            session_id: Session identifier for conversation continuity.
            trace_id: Distributed trace ID for correlation.

        Returns:
            Aggregated results from all concurrently executed sub-agents.
        """
        import asyncio

        # Spawn all sub-agents in parallel
        spawn_tasks: list[Coroutine[object, object, str]] = []
        for task in tasks:
            spawn_task = self.spawner.spawn_subagent(
                parent_agent_id=parent_agent_id,
                task=task,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )
            spawn_tasks.append(spawn_task)

        subagent_ids = await asyncio.gather(*spawn_tasks)

        # Wait for all to complete
        results = await self._wait_for_completion(subagent_ids)

        return self._aggregate_results(results)

    async def _orchestrate_sequential(
        self,
        parent_agent_id: str,
        tasks: list[SpawnTask],
        tenant_id: str | None,
        session_id: str | None,
        trace_id: str | None,
    ) -> OrchestrationResult:
        """Execute sub-agents one at a time, each completing before the next spawns.

        Args:
            parent_agent_id: Identifier of the parent agent.
            tasks: Ordered list of task descriptors to execute sequentially.
            tenant_id: Tenant identifier for data isolation.
            session_id: Session identifier for conversation continuity.
            trace_id: Distributed trace ID for correlation.

        Returns:
            Aggregated results preserving task execution order.
        """
        results: list[SubAgentResult] = []

        for task in tasks:
            subagent_id = await self.spawner.spawn_subagent(
                parent_agent_id=parent_agent_id,
                task=task,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )

            result = await self._wait_for_completion([subagent_id])
            results.append(result[0])

        return self._aggregate_results(results)

    async def _orchestrate_map_reduce(
        self,
        parent_agent_id: str,
        tasks: list[SpawnTask],
        tenant_id: str | None,
        session_id: str | None,
        trace_id: str | None,
    ) -> OrchestrationResult:
        """Execute a two-phase map-reduce orchestration.

        Map phase: spawns sub-agents for all tasks in parallel.
        Reduce phase: spawns a single reducer sub-agent that aggregates map results.

        Args:
            parent_agent_id: Identifier of the parent agent.
            tasks: List of task descriptors for the map phase.
            tenant_id: Tenant identifier for data isolation.
            session_id: Session identifier for conversation continuity.
            trace_id: Distributed trace ID for correlation.

        Returns:
            Aggregated results from the reducer sub-agent.
        """
        import asyncio

        # Map phase: spawn sub-agents for each task
        spawn_tasks: list[Coroutine[object, object, str]] = []
        for task in tasks:
            spawn_task = self.spawner.spawn_subagent(
                parent_agent_id=parent_agent_id,
                task=task,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )
            spawn_tasks.append(spawn_task)

        subagent_ids = await asyncio.gather(*spawn_tasks)

        # Wait for map phase
        map_results = await self._wait_for_completion(subagent_ids)

        # Reduce phase: spawn reducer sub-agent
        reducer_subagent_id = await self.spawner.spawn_subagent(
            parent_agent_id=parent_agent_id,
            task={
                "description": "reduce results",
                "context": {"type": "reduce", "results": map_results},
            },
            tenant_id=tenant_id,
            session_id=session_id,
            trace_id=trace_id,
        )

        # Wait for reducer
        reduce_result = await self._wait_for_completion([reducer_subagent_id])
        return self._aggregate_results(reduce_result)

    async def _wait_for_completion(
        self,
        subagent_ids: list[str],
        timeout: int = 300,
    ) -> list[SubAgentResult]:
        """Poll Worker gRPC for sub-agent completion status.

        Args:
            subagent_ids: List of sub-agent identifiers to monitor.
            timeout: Maximum seconds to wait before declaring a timeout.

        Returns:
            A list of result containers, one per sub-agent.
        """
        # TODO: Implement actual monitoring via Worker gRPC
        # For now, return placeholder
        logger.info(
            "Waiting for %s sub-agents to complete (timeout=%ss)", len(subagent_ids), timeout
        )

        # Placeholder: would call Worker GetSubAgentStatus
        return [
            {
                "subagent_id": subagent_id,
                "status": "completed",
                "result": {},
            }
            for subagent_id in subagent_ids
        ]

    def _aggregate_results(self, results: list[SubAgentResult]) -> OrchestrationResult:
        """Partition sub-agent results into successful and failed buckets.

        Args:
            results: Raw list of results from all executed sub-agents.

        Returns:
            An OrchestrationResult with total count, full list, and partitioned buckets.
        """
        return {
            "total": len(results),
            "results": results,
            "successful": [r for r in results if r.get("status") == "completed"],
            "failed": [r for r in results if r.get("status") == "failed"],
        }
