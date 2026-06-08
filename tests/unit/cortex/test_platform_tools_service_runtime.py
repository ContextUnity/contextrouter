from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from contextunity.core.tokens import ContextToken


def _state(**overrides):
    state = {
        "tenant_id": "state-tenant",
        "__token__": ContextToken(
            token_id="runtime-test",
            user_id="test-user",
            allowed_tenants=("token-tenant",),
            permissions=("worker:execute", "shield:scan"),
        ),
    }
    state.update(overrides)
    return state


@pytest.mark.asyncio
async def test_shield_scan_executor_marks_blocked_result_unsafe():
    from contextunity.router.cortex.compiler.platform_registry import PlatformToolRegistry
    from contextunity.router.cortex.compiler.platform_tools.shield import (
        ShieldScanConfig,
        register_shield_tools,
    )

    registry = PlatformToolRegistry()
    register_shield_tools(registry)
    registration = registry.get("shield_scan")

    mock_client = AsyncMock()
    mock_client.scan.return_value = {
        "allowed": False,
        "blocked": True,
        "reason": "prompt injection",
    }

    with patch(
        "contextunity.router.cortex.compiler.platform_tools.shield._get_shield_client",
        return_value=mock_client,
    ):
        result = await registration.executor(_state(final_output="bad prompt"), ShieldScanConfig())

    assert result["safe"] is False
    assert result["scan_result"]["blocked"] is True
    mock_client.scan.assert_awaited_once_with(content="bad prompt", categories=[])


@pytest.mark.asyncio
async def test_shield_scan_executor_raises_on_error_payload():
    from contextunity.core.exceptions import PlatformServiceError

    from contextunity.router.cortex.compiler.platform_registry import PlatformToolRegistry
    from contextunity.router.cortex.compiler.platform_tools.shield import (
        ShieldScanConfig,
        register_shield_tools,
    )

    registry = PlatformToolRegistry()
    register_shield_tools(registry)
    registration = registry.get("shield_scan")

    mock_client = AsyncMock()
    mock_client.scan.return_value = {"error": "unavailable", "message": "Shield offline"}

    with patch(
        "contextunity.router.cortex.compiler.platform_tools.shield._get_shield_client",
        return_value=mock_client,
    ):
        with pytest.raises(PlatformServiceError, match="Shield offline"):
            await registration.executor(_state(final_output="text"), ShieldScanConfig())


@pytest.mark.asyncio
async def test_worker_start_workflow_executor_passes_token_tenant_and_payload():
    from contextunity.router.cortex.compiler.platform_tools.worker import (
        WorkerStartWorkflowConfig,
        _worker_start_workflow_executor,
    )

    mock_client = AsyncMock()
    mock_client.start_workflow.return_value = {
        "workflow_id": "wf-1",
        "status": "started",
    }

    with patch(
        "contextunity.router.cortex.compiler.platform_tools.worker._get_worker_client",
        return_value=mock_client,
    ):
        result = await _worker_start_workflow_executor(
            _state(),
            WorkerStartWorkflowConfig(
                workflow_type="demo",
                task_queue="tenant-tasks",
                timeout_seconds=42,
                args={"job": 7},
            ),
        )

    mock_client.start_workflow.assert_awaited_once_with(
        workflow_type="demo",
        task_queue="tenant-tasks",
        args=[{"job": 7}],
        timeout_seconds=42,
        provenance=["router:platform_tool:worker_start_workflow"],
    )
    assert result == {"workflow_id": "wf-1", "started": True}


@pytest.mark.asyncio
async def test_worker_get_status_executor_uses_token_tenant_and_state_workflow_id():
    from contextunity.router.cortex.compiler.platform_tools.worker import (
        WorkerGetStatusConfig,
        _worker_get_status_executor,
    )

    mock_client = AsyncMock()
    mock_client.get_task_status.return_value = {"status": "running"}

    with patch(
        "contextunity.router.cortex.compiler.platform_tools.worker._get_worker_client",
        return_value=mock_client,
    ):
        result = await _worker_get_status_executor(
            _state(workflow_id="wf-2"),
            WorkerGetStatusConfig(),
        )

    mock_client.get_task_status.assert_awaited_once_with(workflow_id="wf-2")
    assert result == {"status": "running"}


@pytest.mark.asyncio
async def test_worker_register_schedules_executor_builds_live_schedule_contract():
    from contextunity.router.cortex.compiler.platform_tools.worker import (
        WorkerRegisterSchedulesConfig,
        _worker_register_schedules_executor,
    )

    mock_client = AsyncMock()
    mock_client.register_schedules.return_value = {
        "status": "ok",
        "registered_count": 1,
    }

    with patch(
        "contextunity.router.cortex.compiler.platform_tools.worker._get_worker_client",
        return_value=mock_client,
    ):
        result = await _worker_register_schedules_executor(
            _state(),
            WorkerRegisterSchedulesConfig(
                schedule_name="nightly",
                cron_expression="0 0 * * *",
                workflow_type="sync",
            ),
        )

    mock_client.register_schedules.assert_awaited_once_with(
        project_id="token-tenant",
        schedules=[
            {
                "schedule_id": "nightly",
                "workflow_name": "sync",
                "workflow_class": None,
                "task_queue": "token-tenant-tasks",
                "cron": "0 0 * * *",
            }
        ],
        provenance=["router:platform_tool:worker_register_schedules"],
    )
    assert result == {"registered": True, "registered_count": 1}


@pytest.mark.asyncio
async def test_worker_execute_code_executor_passes_code_contract():
    from contextunity.router.cortex.compiler.platform_tools.worker import (
        WorkerExecuteCodeConfig,
        _worker_execute_code_executor,
    )

    mock_client = AsyncMock()
    mock_client.execute_code.return_value = {"stdout": "ok"}

    with patch(
        "contextunity.router.cortex.compiler.platform_tools.worker._get_worker_client",
        return_value=mock_client,
    ):
        result = await _worker_execute_code_executor(
            _state(final_output="print('ok')"),
            WorkerExecuteCodeConfig(language="python", timeout_seconds=12, sandbox=True),
        )

    mock_client.execute_code.assert_awaited_once_with(
        code="print('ok')",
        language="python",
        timeout_seconds=12,
        sandbox=True,
    )
    assert result == {"stdout": "ok"}
