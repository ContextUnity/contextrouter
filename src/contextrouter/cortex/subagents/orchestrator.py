"""Sub-Agent Orchestrator for Router."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal

from .spawner import SubAgentSpawner

logger = logging.getLogger(__name__)


class SubAgentOrchestrator:
    """Orchestrates sub-agents."""

    def __init__(self):
        self.spawner = SubAgentSpawner()

    async def orchestrate_subagents(
        self,
        parent_agent_id: str,
        tasks: List[Dict[str, Any]],
        strategy: Literal["parallel", "sequential", "map_reduce"] = "parallel",
        tenant_id: str | None = None,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> Dict[str, Any]:
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

        elif strategy == "sequential":
            return await self._orchestrate_sequential(
                parent_agent_id=parent_agent_id,
                tasks=tasks,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )

        elif strategy == "map_reduce":
            return await self._orchestrate_map_reduce(
                parent_agent_id=parent_agent_id,
                tasks=tasks,
                tenant_id=tenant_id,
                session_id=session_id,
                trace_id=trace_id,
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def _orchestrate_parallel(
        self,
        parent_agent_id: str,
        tasks: List[Dict[str, Any]],
        tenant_id: str | None,
        session_id: str | None,
        trace_id: str | None,
    ) -> Dict[str, Any]:
        """Orchestrate sub-agents in parallel."""
        import asyncio

        # Spawn all sub-agents in parallel
        spawn_tasks = []
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
        tasks: List[Dict[str, Any]],
        tenant_id: str | None,
        session_id: str | None,
        trace_id: str | None,
    ) -> Dict[str, Any]:
        """Orchestrate sub-agents sequentially."""
        results = []

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
        tasks: List[Dict[str, Any]],
        tenant_id: str | None,
        session_id: str | None,
        trace_id: str | None,
    ) -> Dict[str, Any]:
        """Orchestrate sub-agents with map-reduce pattern."""
        import asyncio

        # Map phase: spawn sub-agents for each task
        spawn_tasks = []
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
                "type": "reduce",
                "results": map_results,
            },
            tenant_id=tenant_id,
            session_id=session_id,
            trace_id=trace_id,
        )

        # Wait for reducer
        reduce_result = await self._wait_for_completion([reducer_subagent_id])
        return reduce_result[0]

    async def _wait_for_completion(
        self,
        subagent_ids: List[str],
        timeout: int = 300,
    ) -> List[Dict[str, Any]]:
        """Wait for sub-agents to complete."""
        # TODO: Implement actual monitoring via Worker gRPC
        # For now, return placeholder
        logger.info("Waiting for %s sub-agents to complete", len(subagent_ids))

        # Placeholder: would call Worker GetSubAgentStatus
        return [
            {
                "subagent_id": subagent_id,
                "status": "completed",
                "result": {},
            }
            for subagent_id in subagent_ids
        ]

    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple sub-agents."""
        return {
            "total": len(results),
            "results": results,
            "successful": [r for r in results if r.get("status") == "completed"],
            "failed": [r for r in results if r.get("status") == "failed"],
        }
