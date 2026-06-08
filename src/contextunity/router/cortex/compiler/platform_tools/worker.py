"""Worker Platform Tools — workflow executors for compiled graph nodes.

Registers worker_start_workflow, worker_get_status, worker_execute_code,
worker_register_schedules into PlatformToolRegistry.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from contextunity.core import get_contextunit_logger
from contextunity.core.types import JsonDict
from pydantic import BaseModel, ConfigDict, Field

from .helpers.base import resolve_tenant_from_state
from .helpers.contracts import PlatformResult, PlatformState
from .helpers.registration import PlatformRegistry, ToolRegistrationSpec, register_tool_specs
from .helpers.state import get_text_from_state

logger = get_contextunit_logger(__name__)


# ── Config Schemas ──────────────────────────────────────────────────


class WorkerStartWorkflowConfig(BaseModel, frozen=True):
    """Config for worker_start_workflow tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    workflow_type: str
    task_queue: str = "default"
    timeout_seconds: int = Field(default=300, ge=1, le=3600)
    args: JsonDict = Field(default_factory=dict)


class WorkerGetStatusConfig(BaseModel, frozen=True):
    """Config for worker_get_status tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    workflow_id: str = ""


class WorkerExecuteCodeConfig(BaseModel, frozen=True):
    """Config for worker_execute_code tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    language: Literal["python", "javascript", "typescript"] = "python"
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    sandbox: bool = True


class WorkerRegisterSchedulesConfig(BaseModel, frozen=True):
    """Config for worker_register_schedules tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    schedule_name: str = ""
    cron_expression: str = ""
    workflow_type: str = ""


# ── Executor Functions ──────────────────────────────────────────────


def _get_worker_client(tenant_id: str):
    """Get WorkerClient for a tenant."""
    from contextunity.core.sdk import WorkerClient

    from contextunity.router.core.brain_token import get_brain_service_token

    return WorkerClient(
        tenant_id=tenant_id,
        token=get_brain_service_token(allowed_tenants=(tenant_id,)),
    )


async def _worker_start_workflow_executor(
    state: PlatformState, config: WorkerStartWorkflowConfig
) -> PlatformResult:
    """Start a Temporal workflow."""
    tenant_id = resolve_tenant_from_state(state, binding="worker_start_workflow")

    client = _get_worker_client(tenant_id)
    result = await client.start_workflow(
        workflow_type=config.workflow_type,
        task_queue=config.task_queue,
        args=[config.args] if config.args else [],
        timeout_seconds=config.timeout_seconds,
        provenance=["router:platform_tool:worker_start_workflow"],
    )
    payload = dict(result)
    return {
        "workflow_id": payload.get("workflow_id", ""),
        "started": payload.get("status") == "started",
    }


async def _worker_get_status_executor(
    state: PlatformState, config: WorkerGetStatusConfig
) -> PlatformResult:
    """Get workflow execution status."""
    tenant_id = resolve_tenant_from_state(state, binding="worker_get_status")
    workflow_id = config.workflow_id or state.get("workflow_id", "")

    client = _get_worker_client(tenant_id)
    result = await client.get_task_status(workflow_id=workflow_id)
    return dict(result)


async def _worker_execute_code_executor(
    state: PlatformState, config: WorkerExecuteCodeConfig
) -> PlatformResult:
    """Execute code in a sandboxed environment."""
    tenant_id = resolve_tenant_from_state(state, binding="worker_execute_code")
    code = get_text_from_state(state, "code", fallback_key="final_output")

    client = _get_worker_client(tenant_id)
    result = await client.execute_code(
        code=str(code),
        language=config.language,
        timeout_seconds=config.timeout_seconds,
        sandbox=config.sandbox,
    )
    return dict(result)


async def _worker_register_schedules_executor(
    state: PlatformState, config: WorkerRegisterSchedulesConfig
) -> PlatformResult:
    """Register a recurring workflow schedule."""
    tenant_id = resolve_tenant_from_state(state, binding="worker_register_schedules")

    client = _get_worker_client(tenant_id)
    result = await client.register_schedules(
        project_id=tenant_id,
        schedules=[
            {
                "schedule_id": config.schedule_name,
                "workflow_name": config.workflow_type,
                "workflow_class": None,
                "task_queue": f"{tenant_id}-tasks",
                "cron": config.cron_expression,
            }
        ],
        provenance=["router:platform_tool:worker_register_schedules"],
    )
    payload = dict(result)
    return {
        "registered": payload.get("status") == "ok",
        "registered_count": payload.get("registered_count", 0),
    }


# ── Registration ────────────────────────────────────────────────────


def register_worker_tools(registry: PlatformRegistry) -> None:
    """Register all Worker tools into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="worker_start_workflow",
                executor=_worker_start_workflow_executor,
                config_schema=WorkerStartWorkflowConfig,
                required_scopes=["worker:execute"],
            ),
            ToolRegistrationSpec(
                binding="worker_get_status",
                executor=_worker_get_status_executor,
                config_schema=WorkerGetStatusConfig,
                required_scopes=["worker:read"],
            ),
            ToolRegistrationSpec(
                binding="worker_execute_code",
                executor=_worker_execute_code_executor,
                config_schema=WorkerExecuteCodeConfig,
                required_scopes=["worker:execute"],
            ),
            ToolRegistrationSpec(
                binding="worker_register_schedules",
                executor=_worker_register_schedules_executor,
                config_schema=WorkerRegisterSchedulesConfig,
                required_scopes=["worker:execute"],
            ),
        ],
    )


__all__ = ["register_worker_tools", "WorkerStartWorkflowConfig"]
