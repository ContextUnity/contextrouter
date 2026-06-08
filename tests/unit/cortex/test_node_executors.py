import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextunity.core.exceptions import PlatformServiceError, SecurityError
from contextunity.core.tokens import ContextToken

from contextunity.router.core.exceptions import RouterToolTimeout
from contextunity.router.cortex.compiler.node_executors.agent import make_agent_node
from contextunity.router.cortex.compiler.node_executors.federated import (
    make_federated_node,
)
from contextunity.router.cortex.compiler.node_executors.llm import make_llm_node
from contextunity.router.cortex.compiler.node_executors.platform import (
    make_platform_node,
)


def _make_state(**kwargs):
    """Fixture to generate generic state for testing node executors."""
    state = {
        "tenant_id": "test_tenant",
        "messages": [],
        "__token__": ContextToken(
            token_id="test-executor",
            user_id="test-user",
            allowed_tenants=("test_tenant",),
            permissions=("tool:*",),
        ),
        "metadata": {
            "project_id": "test_project",
            "project_config": {"project_id": "test_project"},
        },
    }
    state.update(kwargs)
    return state


@pytest.mark.asyncio
async def test_agent_node_global_goal_prompt_ref_injected():
    """Agent mode injects manifest goal as system prompt before tool calls."""
    node_spec = {
        "name": "my_agent",
        "type": "agent",
        "model": "openai/gpt-5-mini",
        "tools": ["federated:medical_sql"],
        "prompt_ref": "yes",
        "config": {"state_output_key": "answer", "tool_choice": "auto"},
    }
    manifest = {
        "goal": "Solve the user's request using only approved tools.",
        "config": {"my_agent_prompt": "extra line from manifest"},
    }

    mock_tool = MagicMock()
    mock_tool.name = "medical_sql"

    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = []

    mock_bound = AsyncMock()
    mock_bound.ainvoke = AsyncMock(return_value=mock_response)

    from langchain_core.language_models.chat_models import BaseChatModel

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_bound

    with (
        patch(
            "contextunity.router.modules.models.model_registry.create_llm",
            return_value=mock_model,
        ),
        patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=[mock_tool],
        ),
    ):
        executor = make_agent_node(node_spec, manifest)
        await executor(_make_state(messages=[{"role": "user", "content": "hi"}]), {})

    call_msgs = mock_bound.ainvoke.call_args[0][0]
    assert call_msgs[0].type == "system"
    assert "Goal:" in call_msgs[0].content
    assert "Solve the user's request using only approved tools." in call_msgs[0].content
    assert "extra line from manifest" in call_msgs[0].content


@pytest.mark.asyncio
async def test_agent_node_goal_overrides_graph_goal():
    node_spec = {
        "name": "priority_agent",
        "type": "agent",
        "model": "openai/gpt-5-mini",
        "tools": ["federated:medical_sql"],
        "goal": "node-level goal",
        "config": {"state_output_key": "answer", "tool_choice": "auto"},
    }
    manifest = {"goal": "graph-level goal"}

    mock_tool = MagicMock()
    mock_tool.name = "medical_sql"
    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = []
    mock_bound = AsyncMock()
    mock_bound.ainvoke = AsyncMock(return_value=mock_response)
    from langchain_core.language_models.chat_models import BaseChatModel

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_bound

    with (
        patch(
            "contextunity.router.modules.models.model_registry.create_llm",
            return_value=mock_model,
        ),
        patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=[mock_tool],
        ),
    ):
        executor = make_agent_node(node_spec, manifest)
        await executor(_make_state(messages=[{"role": "user", "content": "hi"}]), {})

    sys_msg = mock_bound.ainvoke.call_args[0][0][0]
    assert "node-level goal" in sys_msg.content
    assert "graph-level goal" not in sys_msg.content


@pytest.mark.asyncio
async def test_agent_node_default_goal_when_manifest_omits_goal():
    node_spec = {
        "name": "default_goal_agent",
        "type": "agent",
        "model": "openai/gpt-5-mini",
        "tools": ["federated:medical_sql"],
        "config": {"state_output_key": "answer", "tool_choice": "auto"},
    }

    mock_tool = MagicMock()
    mock_tool.name = "medical_sql"
    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = []
    mock_bound = AsyncMock()
    mock_bound.ainvoke = AsyncMock(return_value=mock_response)
    from langchain_core.language_models.chat_models import BaseChatModel

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_bound

    with (
        patch(
            "contextunity.router.modules.models.model_registry.create_llm",
            return_value=mock_model,
        ),
        patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=[mock_tool],
        ),
    ):
        executor = make_agent_node(node_spec, {})
        await executor(_make_state(messages=[{"role": "user", "content": "hi"}]), {})

    sys_msg = mock_bound.ainvoke.call_args[0][0][0]
    assert sys_msg.type == "system"
    assert "Goal:" in sys_msg.content
    assert "approved tools available to this agent" in sys_msg.content


@pytest.mark.asyncio
async def test_agent_node_system_prompt_extends_goal_not_persona():
    node_spec = {
        "name": "ov_agent",
        "type": "agent",
        "model": "openai/gpt-5-mini",
        "tools": ["federated:medical_sql"],
        "prompt_ref": "yes",
        "config": {
            "state_output_key": "answer",
            "tool_choice": "auto",
            "system_prompt": "explicit override",
        },
    }
    manifest = {
        "persona": "p1",
        "goal": "global goal",
        "config": {"ov_agent_prompt": "appended ref"},
    }

    mock_tool = MagicMock()
    mock_tool.name = "medical_sql"
    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = []
    mock_bound = AsyncMock()
    mock_bound.ainvoke = AsyncMock(return_value=mock_response)
    from langchain_core.language_models.chat_models import BaseChatModel

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_bound

    with (
        patch(
            "contextunity.router.modules.models.model_registry.create_llm",
            return_value=mock_model,
        ),
        patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=[mock_tool],
        ),
    ):
        executor = make_agent_node(node_spec, manifest)
        await executor(_make_state(messages=[{"role": "user", "content": "hi"}]), {})

    sys_msg = mock_bound.ainvoke.call_args[0][0][0]
    assert "Goal:" in sys_msg.content
    assert "global goal" in sys_msg.content
    assert "explicit override" in sys_msg.content
    assert "appended ref" in sys_msg.content
    assert "p1" not in sys_msg.content


@pytest.mark.asyncio
async def test_agent_node_ignores_persona_in_agent_mode():
    """Agent mode uses goal; persona remains LLM-node behavior only."""
    node_spec = {
        "name": "agent_no_persona",
        "type": "agent",
        "model": "openai/gpt-5-mini",
        "tools": ["federated:medical_sql"],
        "config": {"state_output_key": "answer", "tool_choice": "auto"},
    }
    manifest = {"persona": "legacy-persona", "goal": "agent goal"}

    mock_tool = MagicMock()
    mock_tool.name = "medical_sql"
    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = []
    mock_bound = AsyncMock()
    mock_bound.ainvoke = AsyncMock(return_value=mock_response)
    from langchain_core.language_models.chat_models import BaseChatModel

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_bound

    with (
        patch(
            "contextunity.router.modules.models.model_registry.create_llm",
            return_value=mock_model,
        ),
        patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=[mock_tool],
        ),
    ):
        executor = make_agent_node(node_spec, manifest)
        await executor(_make_state(messages=[{"role": "user", "content": "hi"}]), {})

    sys_msg = mock_bound.ainvoke.call_args[0][0][0]
    assert "agent goal" in sys_msg.content
    assert "legacy-persona" not in sys_msg.content


@pytest.mark.asyncio
async def test_agent_node_executor_binds_allowed_tools():
    """Agent node uses explicit tools and keeps tool execution on SecureTool path."""
    node_spec = {
        "name": "agent",
        "type": "agent",
        "model": "openai/gpt-5-mini",
        "tools": ["federated:medical_sql"],
        "config": {"state_output_key": "answer", "tool_choice": "auto"},
    }

    mock_tool = MagicMock()
    mock_tool.name = "medical_sql"

    mock_response = MagicMock()
    mock_response.content = "done"
    mock_response.tool_calls = []

    mock_bound = AsyncMock()
    mock_bound.ainvoke.return_value = mock_response

    from langchain_core.language_models.chat_models import BaseChatModel

    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools.return_value = mock_bound

    with (
        patch(
            "contextunity.router.modules.models.model_registry.create_llm",
            return_value=mock_model,
        ),
        patch(
            "contextunity.router.modules.tools.discover_all_tools",
            return_value=[mock_tool],
        ),
    ):
        executor = make_agent_node(node_spec, {})
        result = await executor(_make_state(messages=[{"role": "user", "content": "hi"}]), {})

    assert result["answer"] == "done"
    mock_model.bind_tools.assert_called_once_with([mock_tool], tool_choice="auto")


@pytest.mark.asyncio
async def test_llm_node_executor():
    """Test LLM Node execution and output key routing."""
    node_spec = {
        "name": "analyze_intent",
        "type": "llm",
        "config": {
            "model": "gpt-4o",
            "prompt_ref": "some_prompt",
            "state_output_key": "classification",
        },
    }

    from contextunity.router.modules.models.types import ModelResponse, ProviderInfo

    with patch(
        "contextunity.router.cortex.compiler.node_executors.llm._get_model_registry"
    ) as mock_get_registry:
        mock_model = AsyncMock()
        mock_model.generate.return_value = ModelResponse(
            text="result_content",
            raw_provider=ProviderInfo(
                provider="openai", model_name="gpt-4o", model_key="openai/gpt-4o"
            ),
        )
        mock_get_registry.return_value.create_llm.return_value = mock_model

        executor = make_llm_node(node_spec, {})
        result = await executor(_make_state(messages=[{"role": "user", "content": "test"}]), {})

        # The output should route to the config-specified 'state_output_key'
        assert "classification" in result
        assert result["classification"] == "result_content"
        mock_get_registry.return_value.create_llm.assert_called_once()
        mock_model.generate.assert_called_once()


@pytest.mark.asyncio
async def test_federated_node_executor():
    """Test Federated Node execution and tagging."""
    node_spec = {
        "name": "fetch_custom_data",
        "type": "tool",
        "tool_binding": "federated:user_custom_tool",
        "config": {
            "timeout": 10,
            "state_input_key": "some_query",
            "state_output_key": "products",
        },
    }

    with patch(
        "contextunity.router.cortex.compiler.node_executors.federated._get_stream_executor_manager"
    ) as mock_get_manager:
        mock_manager = MagicMock()
        mock_manager.execute = AsyncMock(return_value={"fetched": 123})
        mock_get_manager.return_value = mock_manager

        executor = make_federated_node(node_spec, {})
        result = await executor(_make_state(some_query="find things"), {})

        assert "products" in result
        output = result["products"]
        # Output MUST be tagged as untrusted
        assert output.get("__untrusted__") is True
        assert output.get("__source__") == "federated:user_custom_tool"
        assert output.get("fetched") == 123
        mock_manager.execute.assert_awaited()
        assert mock_manager.execute.await_args.kwargs["project_id"] == "test_project"


@pytest.mark.asyncio
async def test_federated_node_timeout():
    """Test Federated Node timeout handling — wraps TimeoutError as RouterToolTimeout."""
    node_spec = {
        "name": "slow_tool",
        "tool_binding": "federated:slow_action",
        "config": {"timeout": 5},
    }

    with patch(
        "contextunity.router.cortex.compiler.node_executors.federated._get_stream_executor_manager"
    ) as mock_get_manager:
        mock_manager = MagicMock()
        mock_manager.execute = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_get_manager.return_value = mock_manager

        executor = make_federated_node(node_spec, {})

        with pytest.raises(RouterToolTimeout) as exc:
            await executor(_make_state(), {})

        assert exc.value.details.get("node_name") == "slow_tool"
        assert exc.value.details.get("tool_binding") == "federated:slow_action"


@pytest.mark.asyncio
async def test_platform_node_executor():
    """Test Platform Node execution configuration parsing."""
    node_spec = {
        "name": "save_to_memory",
        "type": "tool",
        "tool_binding": "brain_blackboard_write",
        "config": {
            "state_input_key": "classification",
        },
    }

    executor = make_platform_node(node_spec, {})

    # With lazy-init, brain_blackboard_write IS registered.
    # Without a token, the executor hits the scope check → SecurityError.
    with pytest.raises(SecurityError):
        await executor(_make_state(classification="test payload"), {})


@pytest.mark.asyncio
async def test_federated_node_non_timeout_error():
    """Non-timeout federated errors raise PlatformServiceError, not RouterToolTimeout."""
    node_spec = {
        "name": "broken_tool",
        "tool_binding": "federated:broken_action",
        "config": {"timeout": 5},
    }

    with (
        patch(
            "contextunity.router.cortex.compiler.node_executors.federated._get_stream_executor_manager"
        ) as mock_get_manager,
        patch(
            "contextunity.router.cortex.compiler.node_executors.federated.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        mock_manager = MagicMock()
        mock_manager.execute = AsyncMock(side_effect=ConnectionError("connection refused"))
        mock_get_manager.return_value = mock_manager

        executor = make_federated_node(node_spec, {})

        with pytest.raises(PlatformServiceError) as exc:
            await executor(_make_state(), {})

        assert exc.value.details.get("node_name") == "broken_tool"
        # Must NOT be RouterToolTimeout
        assert not isinstance(exc.value, RouterToolTimeout)


# ============================================================================
# Phase 1.4: AG-UI BrainEvent emission tests
# ============================================================================


@pytest.mark.asyncio
async def test_platform_executor_emits_tool_result_ok_event():
    """Platform executor emits unified tool_result event on success."""
    node_spec = {
        "name": "search_brain",
        "type": "tool",
        "tool_binding": "brain_search",
        "tool_name": "brain_search",
        "tool_kind": "platform",
        "meta": {"handler": "brain_search", "source": "platform_registry"},
        "config": {},
    }

    mock_registry = MagicMock()
    mock_registration = MagicMock()
    mock_registration.executor = AsyncMock(return_value={"results": []})
    mock_registry.get.return_value = mock_registration
    mock_registry.validate_config.return_value = {}

    dispatched_events = []

    def capture_event(name, data):
        dispatched_events.append((name, data))

    with (
        patch(
            "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
            return_value=mock_registry,
        ),
        patch(
            "contextunity.router.cortex.compiler.node_executors.telemetry.dispatch_custom_event",
            side_effect=capture_event,
        ),
    ):
        executor = make_platform_node(node_spec, {})
        await executor(_make_state(), {})

    result_events = [e for e in dispatched_events if e[1].get("event", {}).type == "tool_result"]
    assert len(result_events) == 1, f"Expected 1 tool_result event, got {len(result_events)}"
    event = result_events[0][1]["event"]
    assert event.node == "search_brain"
    assert event.data.get("status") == "ok"
    assert event.data.get("tool_binding") == "brain_search"
    assert isinstance(event.data.get("duration_ms"), int)


@pytest.mark.asyncio
async def test_platform_executor_tool_result_contains_runtime_metadata():
    """tool_result includes handler/source/toolkit fields for traces."""
    node_spec = {
        "name": "search_brain",
        "type": "tool",
        "tool_binding": "brain_search",
        "tool_name": "brain_search",
        "tool_kind": "platform",
        "meta": {
            "handler": "brain_search",
            "source": "platform_registry",
            "toolkit": "PlatformBuiltin",
        },
        "config": {},
    }

    mock_registry = MagicMock()
    mock_registration = MagicMock()
    mock_registration.executor = AsyncMock(return_value={"results": []})
    mock_registry.get.return_value = mock_registration
    mock_registry.validate_config.return_value = {}

    dispatched_events = []

    def capture_event(name, data):
        dispatched_events.append((name, data))

    with (
        patch(
            "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
            return_value=mock_registry,
        ),
        patch(
            "contextunity.router.cortex.compiler.node_executors.telemetry.dispatch_custom_event",
            side_effect=capture_event,
        ),
    ):
        executor = make_platform_node(node_spec, {})
        await executor(_make_state(), {})

    result_events = [e for e in dispatched_events if e[1].get("event", {}).type == "tool_result"]
    assert len(result_events) == 1, f"Expected 1 tool_result event, got {len(result_events)}"
    data = result_events[0][1]["event"].data
    assert data.get("handler") == "brain_search"
    assert data.get("source") == "platform_registry"
    assert data.get("toolkit") == "PlatformBuiltin"


@pytest.mark.asyncio
async def test_platform_executor_emits_tool_error_on_failure():
    """Platform executor must emit tool_result(error) BrainEvent on failure."""
    node_spec = {
        "name": "broken_platform",
        "type": "tool",
        "tool_binding": "brain_search",
        "tool_name": "brain_search",
        "tool_kind": "platform",
        "config": {},
    }

    mock_registry = MagicMock()
    mock_registration = MagicMock()
    mock_registration.executor = AsyncMock(
        side_effect=PlatformServiceError(
            message="service down", node_name="broken_platform", tool_binding="brain_search"
        )
    )
    mock_registry.get.return_value = mock_registration
    mock_registry.validate_config.return_value = {}

    dispatched_events = []

    def capture_event(name, data):
        dispatched_events.append((name, data))

    with (
        patch(
            "contextunity.router.cortex.compiler.node_executors.platform._get_platform_registry",
            return_value=mock_registry,
        ),
        patch(
            "contextunity.router.cortex.compiler.node_executors.telemetry.dispatch_custom_event",
            side_effect=capture_event,
        ),
    ):
        executor = make_platform_node(node_spec, {})
        with pytest.raises(PlatformServiceError):
            await executor(_make_state(), {})

    result_events = [e for e in dispatched_events if e[1].get("event", {}).type == "tool_result"]
    assert len(result_events) == 1, f"Expected 1 tool_result event, got {len(result_events)}"
    event = result_events[0][1]["event"]
    assert event.node == "broken_platform"
    assert event.data.get("status") == "error"
    assert event.data.get("error", {}).get("code") == "platform_service_error"


@pytest.mark.asyncio
async def test_llm_executor_emits_llm_error_on_failure():
    """LLM executor must emit llm_error BrainEvent when LLM invocation fails."""
    from contextunity.router.core.exceptions import RouterLLMError

    node_spec = {
        "name": "broken_llm",
        "type": "llm",
        "model": "gpt-4o",
        "config": {"state_output_key": "output"},
    }

    dispatched_events = []

    def capture_event(name, data):
        dispatched_events.append((name, data))

    with (
        patch(
            "contextunity.router.cortex.compiler.node_executors.llm._get_model_registry"
        ) as mock_get_registry,
        patch(
            "contextunity.router.cortex.compiler.node_executors.llm.dispatch_custom_event",
            side_effect=capture_event,
        ),
    ):
        mock_model = AsyncMock()
        mock_model.generate.side_effect = Exception("LLM crashed")
        mock_get_registry.return_value.create_llm.return_value = mock_model

        executor = make_llm_node(node_spec, {})
        with pytest.raises(RouterLLMError):
            await executor(_make_state(messages=[{"role": "user", "content": "test"}]), {})

    llm_error_events = [e for e in dispatched_events if e[1].get("event", {}).type == "llm_error"]
    assert len(llm_error_events) == 1, f"Expected 1 llm_error event, got {len(llm_error_events)}"
    assert llm_error_events[0][1]["event"].node == "broken_llm"


# ── LLM Node Factory — Mutation Killers ───────────────────────────────────


@pytest.mark.asyncio
async def test_llm_node_parallel_mode_raises():
    """mode='parallel' is reserved — must raise NotImplementedError.

    Kills mutant: NotImplementedError removal.
    """
    node_spec = {"name": "par_node", "mode": "parallel"}
    with pytest.raises(NotImplementedError, match="parallel"):
        make_llm_node(node_spec, {})


@pytest.mark.asyncio
async def test_llm_node_default_output_key():
    """Default state_output_key is 'final_output' when not specified in config.

    Kills mutant: state_output_key default mutation.
    """
    node_spec = {
        "name": "default_key_node",
        "type": "llm",
        "config": {},
    }

    from contextunity.router.modules.models.types import ModelResponse, ProviderInfo

    with patch(
        "contextunity.router.cortex.compiler.node_executors.llm._get_model_registry"
    ) as mock_get_registry:
        mock_model = AsyncMock()
        mock_model.generate.return_value = ModelResponse(
            text="output",
            raw_provider=ProviderInfo(
                provider="openai", model_name="gpt-4o", model_key="openai/gpt-4o"
            ),
        )
        mock_get_registry.return_value.create_llm.return_value = mock_model

        executor = make_llm_node(node_spec, {})
        result = await executor(_make_state(messages=[{"role": "user", "content": "x"}]), {})

        assert "final_output" in result, "Default state_output_key must be 'final_output'"


@pytest.mark.asyncio
async def test_llm_node_json_output_format():
    """output_format='json' parses LLM output as JSON and merges with input.

    Kills mutant: output_format branch removal.
    """
    node_spec = {
        "name": "json_node",
        "type": "llm",
        "config": {
            "output_format": "json",
            "state_input_key": "messages",
            "state_output_key": "parsed",
        },
    }

    from contextunity.router.modules.models.types import ModelResponse, ProviderInfo

    with patch(
        "contextunity.router.cortex.compiler.node_executors.llm._get_model_registry"
    ) as mock_get_registry:
        mock_model = AsyncMock()
        mock_model.generate.return_value = ModelResponse(
            text='{"valid": true, "reason": "looks good"}',
            raw_provider=ProviderInfo(
                provider="openai", model_name="gpt-4o", model_key="openai/gpt-4o"
            ),
        )
        mock_get_registry.return_value.create_llm.return_value = mock_model

        executor = make_llm_node(node_spec, {})
        result = await executor(_make_state(messages=[{"role": "user", "content": "x"}]), {})

        assert "parsed" in result
        assert result["parsed"]["valid"] is True
        assert "intermediate_results" in result


@pytest.mark.asyncio
async def test_llm_node_records_last_node():
    """_last_node key records the node name for downstream routing.

    Kills mutant: _last_node assignment removal.
    """
    node_spec = {
        "name": "tracked_node",
        "type": "llm",
        "config": {},
    }

    from contextunity.router.modules.models.types import ModelResponse, ProviderInfo

    with patch(
        "contextunity.router.cortex.compiler.node_executors.llm._get_model_registry"
    ) as mock_get_registry:
        mock_model = AsyncMock()
        mock_model.generate.return_value = ModelResponse(
            text="result",
            raw_provider=ProviderInfo(
                provider="openai", model_name="gpt-4o", model_key="openai/gpt-4o"
            ),
        )
        mock_get_registry.return_value.create_llm.return_value = mock_model

        executor = make_llm_node(node_spec, {})
        result = await executor(_make_state(messages=[{"role": "user", "content": "x"}]), {})

        assert result["_last_node"] == "tracked_node"
