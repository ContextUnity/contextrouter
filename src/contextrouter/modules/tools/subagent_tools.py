"""Sub-agent spawning tools for Dispatcher Agent."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool

from contextrouter.cortex.runtime_context import get_current_access_token
from contextrouter.cortex.subagents.spawner import SubAgentSpawner
from contextrouter.modules.tools import register_tool

logger = logging.getLogger(__name__)

# Global spawner instance
_spawner: SubAgentSpawner | None = None


def _get_spawner() -> SubAgentSpawner:
    """Get or create spawner instance."""
    global _spawner
    if _spawner is None:
        _spawner = SubAgentSpawner()
    return _spawner


@tool
async def spawn_subagent(
    task: dict[str, Any],
    agent_type: str = "task_executor",
    strategy: str = "sequential",
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Spawn a sub-agent to handle a specific task.

    Use this tool when:
    - A task requires specialized processing that would benefit from isolation
    - A task can be parallelized with other tasks
    - A task needs to run independently from the main agent flow
    - You need to delegate work to a specialized agent type

    Args:
        task: Task description and parameters (dict with task details)
        agent_type: Type of agent to spawn (default: "task_executor")
            - "task_executor": General purpose task executor
            - "parallel_worker": For parallel processing
            - "specialized_agent": For domain-specific tasks
        strategy: Execution strategy (default: "sequential")
            - "sequential": Execute tasks one after another
            - "parallel": Execute tasks in parallel
            - "map_reduce": Map tasks, then reduce results
        tenant_id: Tenant ID for multi-tenant isolation (optional)
        session_id: Session ID for session isolation (optional)

    Returns:
        Dictionary with:
        - subagent_id: ID of the spawned sub-agent
        - status: "spawned" or "error"
        - message: Status message

    Example:
        {
            "task": {
                "description": "Process user data",
                "data": {...}
            },
            "agent_type": "task_executor",
            "strategy": "sequential"
        }
    """
    try:
        spawner = _get_spawner()

        # Extract trace_id from current context if available
        # Note: This will be automatically passed from DispatcherState
        # when the tool is called from within the dispatcher graph
        trace_id = None  # Will be set by dispatcher graph integration
        access_token = get_current_access_token()

        subagent_id = await spawner.spawn_subagent(
            parent_agent_id="dispatcher",
            task=task,
            tenant_id=tenant_id,
            session_id=session_id,
            trace_id=trace_id,
            agent_type=agent_type,
            config={"strategy": strategy},
            token=access_token,
        )

        logger.info(
            "Spawned sub-agent %s for task: %s", subagent_id, task.get("description", "unknown")
        )

        return {
            "subagent_id": subagent_id,
            "status": "spawned",
            "message": f"Sub-agent {subagent_id} spawned successfully",
            "agent_type": agent_type,
            "strategy": strategy,
        }

    except Exception as e:
        logger.exception("Failed to spawn sub-agent: %s", e)
        return {
            "subagent_id": None,
            "status": "error",
            "message": f"Failed to spawn sub-agent: {str(e)}",
            "error": str(e),
        }


# Auto-register the tool when module is imported
_spawn_subagent_tool = spawn_subagent
register_tool(_spawn_subagent_tool)
