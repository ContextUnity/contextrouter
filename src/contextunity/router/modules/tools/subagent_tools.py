"""Sub-agent spawning tools for Dispatcher Agent."""

from __future__ import annotations

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonDict

from contextunity.router.cortex.subagents.spawner import SubAgentSpawner
from contextunity.router.langchain_boundaries import tool
from contextunity.router.modules.tools import register_tool
from contextunity.router.modules.tools.schemas import SubAgentResult

logger = get_contextunit_logger(__name__)

# Spawn limits — prevent runaway recursive delegation and per-session spawn floods.
MAX_SUBAGENT_DEPTH = 3
MAX_SPAWNS_PER_SESSION = 20

# Per-(tenant, session) spawn counters for the budget check.
_spawn_counts: dict[tuple[str, str], int] = {}

# Global spawner instance
_spawner: SubAgentSpawner | None = None


def _subagent_depth(provenance: tuple[str, ...]) -> int:
    """Count delegation hops already taken through subagent spawning.

    TokenBuilder.attenuate appends ``>subagent:{type}`` to provenance on each
    spawn, so the count is the nesting depth of the caller.
    """
    return sum(1 for entry in provenance if entry.lstrip(">").startswith("subagent:"))


def _get_spawner() -> SubAgentSpawner:
    """Get or create spawner instance."""
    global _spawner
    if _spawner is None:
        _spawner = SubAgentSpawner()
    return _spawner


def _tenant_denied(message: str) -> SubAgentResult:
    return SubAgentResult(
        success=False,
        subagent_id=None,
        status="error",
        message=message,
        error="tenant_access_denied",
    )


def _resolve_spawn_tenant(
    tenant_id: str | None, current_token: object | None
) -> str | SubAgentResult:
    """Resolve tenant from caller token; never trust tool args alone."""
    requested = tenant_id.strip() if isinstance(tenant_id, str) and tenant_id.strip() else None
    if current_token is None:
        return _tenant_denied("Spawn refused: missing caller token for tenant-scoped sub-agent")

    has_permission = getattr(current_token, "has_permission", None)
    if callable(has_permission) and has_permission("admin:all"):
        return requested or "default"

    allowed = tuple(str(t) for t in getattr(current_token, "allowed_tenants", ()) if str(t))
    if requested:
        can_access_tenant = getattr(current_token, "can_access_tenant", None)
        if callable(can_access_tenant):
            allowed_requested = bool(can_access_tenant(requested))
        else:
            allowed_requested = requested in allowed
        if allowed_requested:
            return requested
        return _tenant_denied(f"Spawn refused: tenant {requested!r} is outside caller scope")

    if len(allowed) == 1:
        return allowed[0]
    if not allowed:
        return _tenant_denied("Spawn refused: caller token has no tenant scope")
    return _tenant_denied(
        "Spawn refused: tenant_id is required when caller token has multiple tenants"
    )


@tool
async def spawn_subagent(
    task: JsonDict,
    agent_type: str = "task_executor",
    strategy: str = "sequential",
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> SubAgentResult:
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
        from contextunity.core.tokens import TokenBuilder

        from contextunity.router.core.context import get_current_access_token

        current_token = get_current_access_token()
        resolved_tenant = _resolve_spawn_tenant(tenant_id, current_token)
        if not isinstance(resolved_tenant, str):
            return resolved_tenant
        tenant_id = resolved_tenant

        # Depth limit: refuse spawns from callers already nested too deep.
        if current_token is not None:
            depth = _subagent_depth(current_token.provenance)
            if depth >= MAX_SUBAGENT_DEPTH:
                return SubAgentResult(
                    success=False,
                    subagent_id=None,
                    status="error",
                    message=f"Spawn refused: max sub-agent nesting depth {MAX_SUBAGENT_DEPTH} reached",
                    error="max_depth_exceeded",
                )

        # Per-session budget: bound total spawns per (tenant, session).
        budget_key = (tenant_id, session_id or "default")
        spawned = _spawn_counts.get(budget_key, 0)
        if spawned >= MAX_SPAWNS_PER_SESSION:
            return SubAgentResult(
                success=False,
                subagent_id=None,
                status="error",
                message=f"Spawn refused: session budget of {MAX_SPAWNS_PER_SESSION} sub-agents exhausted",
                error="spawn_budget_exhausted",
            )
        _spawn_counts[budget_key] = spawned + 1

        attenuated_token = None
        if current_token:
            attenuated_token = TokenBuilder().attenuate(
                current_token,
                permissions=None,
                agent_id=f"subagent:{agent_type}",
            )

        from contextunity.router.cortex.subagents.types import (
            SpawnTask,
        )
        from contextunity.router.cortex.subagents.types import (
            SubAgentConfig as SubAgentConfigSpec,
        )

        description_raw = task.get("description", "")
        description = str(description_raw) if description_raw is not None else ""
        context_raw = task.get("context", "")
        context = str(context_raw) if context_raw is not None else ""

        subagent_id = await spawner.spawn_subagent(
            parent_agent_id="dispatcher",
            task=SpawnTask(description=description, context=context),
            tenant_id=tenant_id,
            session_id=session_id,
            trace_id=trace_id,
            agent_type=agent_type,
            config=SubAgentConfigSpec(strategy=strategy),
            token=attenuated_token,
        )

        task_label = description or "unknown"
        logger.info("Spawned sub-agent %s for task: %s", subagent_id, task_label)

        return SubAgentResult(
            success=True,
            subagent_id=subagent_id,
            status="spawned",
            message=f"Sub-agent {subagent_id} spawned successfully",
            agent_type=agent_type,
            strategy=strategy,
        )

    except Exception as e:
        logger.exception("Failed to spawn sub-agent: %s", e)
        return SubAgentResult(
            success=False,
            subagent_id=None,
            status="error",
            message=f"Failed to spawn sub-agent: {e!s}",
            error=str(e),
        )


# Auto-register the tool when module is imported
_spawn_subagent_tool = spawn_subagent
register_tool(_spawn_subagent_tool)
