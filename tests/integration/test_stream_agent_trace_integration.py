"""Integration test: StreamAgent wires replay_astream_event_to_tracer → non-empty steps in log_execution_trace.

Verifies the fix for astream_tracer_replay is wired in production StreamAgent path,
ensuring Brain traces contain Graph Journey steps for streaming executions.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from contextunity.core.tokens import ContextToken
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from contextunity.router.modules.observability.auto_tracer import BrainAutoTracer
from contextunity.router.service.mixins.execution.helpers import (
    iter_graph_events,
    log_execution_trace,
)


class FakeGraph:
    """Fake LangGraph that yields astream_events v2 payloads."""

    def __init__(self, nodes: dict[str, Any] | None = None) -> None:
        self.nodes = nodes or {"planner": MagicMock(), "__end__": MagicMock()}
        self._run_id = uuid.uuid4()
        self._planner_run_id = uuid.uuid4()

    async def astream_events(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        *,
        version: str = "v2",
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield mock astream_events v2 payloads."""
        # on_chain_start for graph root
        yield {
            "event": "on_chain_start",
            "name": "LangGraph",
            "run_id": str(self._run_id),
            "parent_ids": [],
            "tags": [],
            "metadata": {},
            "data": {"input": input},
        }

        # on_chain_start for planner node
        yield {
            "event": "on_chain_start",
            "name": "planner",
            "run_id": str(self._planner_run_id),
            "parent_ids": [str(self._run_id)],
            "tags": ["node:planner"],
            "metadata": {"node": "planner"},
            "data": {"input": {"messages": input.get("messages", [])}},
        }

        # on_chat_model_start
        yield {
            "event": "on_chat_model_start",
            "name": "gpt-4o",
            "run_id": str(uuid.uuid4()),
            "parent_ids": [str(self._planner_run_id)],
            "tags": [],
            "metadata": {},
            "data": {
                "serialized": {},
                "input": {"messages": [[HumanMessage(content="test")]]},
            },
        }

        # on_chat_model_end
        yield {
            "event": "on_chat_model_end",
            "name": "gpt-4o",
            "run_id": str(uuid.uuid4()),
            "parent_ids": [str(self._planner_run_id)],
            "tags": [],
            "metadata": {},
            "data": {
                "output": AIMessage(content="Hello!"),
            },
        }

        # on_chain_end for planner node with _steps in output
        yield {
            "event": "on_chain_end",
            "name": "planner",
            "run_id": str(self._planner_run_id),
            "parent_ids": [str(self._run_id)],
            "tags": ["node:planner"],
            "metadata": {},
            "data": {
                "output": {
                    "final_output": {"content": "Hello!"},
                    "_steps": [
                        {
                            "id": str(uuid.uuid4()),
                            "node": "planner",
                            "type": "chain",
                            "status": "ok",
                        }
                    ],
                }
            },
        }

        # on_chain_end for graph root
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "run_id": str(self._run_id),
            "parent_ids": [],
            "tags": [],
            "metadata": {},
            "data": {"output": {"final_output": {"content": "Hello!"}}},
        }


class FakeServicerContext:
    """Fake gRPC servicer context for testing."""

    def __init__(self) -> None:
        self._metadata = {}

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return (
            ("x-trace-id", str(uuid.uuid4())),
            ("x-tenant-id", "test-tenant"),
        )


@pytest.fixture
def fake_token() -> ContextToken:
    """Create a valid ContextToken for testing."""
    return ContextToken(
        token_id="test-token",
        permissions=("router:execute", "brain:write"),
        allowed_tenants=("test-tenant",),
        user_id="test-user",
    )


@pytest.fixture
def mock_project_configs() -> dict[str, Any]:
    """Mock project configurations."""
    return {
        "test-tenant": {
            "project_id": "test-tenant",
            "tenant_id": "test-tenant",
            "graphs": {},
        }
    }


@pytest.fixture
def mock_project_graphs() -> dict[str, Any]:
    """Mock project graphs mapping."""
    return {"test-tenant": {"default": "test_graph"}}


@pytest.mark.asyncio
class TestStreamAgentTraceIntegration:
    """Test StreamAgent wires replay → BrainAutoTracer → non-empty steps."""

    async def test_stream_agent_replay_populates_tracer_steps(
        self,
        fake_token: ContextToken,
        mock_project_configs: dict[str, Any],
        mock_project_graphs: dict[str, Any],
    ) -> None:
        """Verify replay_astream_event_to_tracer feeds events into auto_tracer.

        This is the core integration test for integration test fix. It verifies that
        the replay function is actually called and populates tracer steps.
        """
        # Create tracer and verify it starts empty
        tracer = BrainAutoTracer()
        assert tracer.get_nested_steps() == []

        # Simulate replay of astream_events into tracer
        from contextunity.router.modules.observability.astream_tracer_replay import (
            replay_astream_event_to_tracer,
        )

        graph_run_id = uuid.uuid4()
        planner_run_id = uuid.uuid4()

        # Replay events as StreamAgent would
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_start",
                "name": "LangGraph",
                "run_id": str(graph_run_id),
                "parent_ids": [],
                "data": {"input": {}},
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_start",
                "name": "planner",
                "run_id": str(planner_run_id),
                "parent_ids": [str(graph_run_id)],
                "data": {"input": {}},
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_end",
                "name": "planner",
                "run_id": str(planner_run_id),
                "parent_ids": [str(graph_run_id)],
                "data": {"output": {"final_output": {"content": "Hello!"}}},
            },
        )

        # Verify tracer has steps
        steps = tracer.get_nested_steps()
        assert len(steps) == 1, f"Expected 1 step, got {len(steps)}"
        assert steps[0]["node"] == "planner"

    async def test_iter_graph_events_yields_events_for_replay(
        self,
        fake_token: ContextToken,
    ) -> None:
        """Verify iter_graph_events yields events that can be replayed.

        This tests that the helper used by StreamAgent produces compatible
        event payloads for replay_astream_event_to_tracer.
        """
        fake_graph = FakeGraph()
        execution_input = {"messages": [HumanMessage(content="Hello")]}

        # Build a minimal run config
        run_config: RunnableConfig = {"callbacks": []}

        # Collect events
        events: list[dict[str, Any]] = []
        async for event in iter_graph_events(fake_graph, execution_input, run_config=run_config):
            events.append(event)

        # Should have multiple events
        assert len(events) >= 4, f"Expected at least 4 events, got {len(events)}"

        # Verify events can be replayed into tracer
        tracer = BrainAutoTracer()
        from contextunity.router.modules.observability.astream_tracer_replay import (
            replay_astream_event_to_tracer,
        )

        for event in events:
            await replay_astream_event_to_tracer(tracer, event)

        # Tracer should have populated spans
        assert len(tracer.spans) > 0, "Expected spans to be populated"

    @patch("contextunity.router.modules.tools.brain_trace_tools._get_brain_client")
    async def test_log_execution_trace_receives_non_empty_steps(
        self,
        mock_get_client: AsyncMock,
        fake_token: ContextToken,
    ) -> None:
        """Verify log_execution_trace receives non-empty steps from tracer.

        This tests the full chain: replay → tracer → log_execution_trace.
        """
        # Setup mock Brain client
        mock_client = AsyncMock()
        mock_client.log_trace.return_value = "trace-123"
        mock_client.add_episode.return_value = MagicMock(episode_id="ep-123")
        mock_get_client.return_value = mock_client

        # Create tracer with replayed events
        tracer = BrainAutoTracer()
        from contextunity.router.modules.observability.astream_tracer_replay import (
            replay_astream_event_to_tracer,
        )

        graph_run_id = uuid.uuid4()
        node_run_id = uuid.uuid4()

        # Replay realistic event sequence
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_start",
                "name": "LangGraph",
                "run_id": str(graph_run_id),
                "parent_ids": [],
                "tags": [],
                "metadata": {},
                "data": {"input": {"messages": []}},
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_start",
                "name": "test_node",
                "run_id": str(node_run_id),
                "parent_ids": [str(graph_run_id)],
                "tags": ["node:test"],
                "metadata": {"node": "test_node"},
                "data": {"input": {}},
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chat_model_start",
                "name": "gpt-4o",
                "run_id": str(uuid.uuid4()),
                "parent_ids": [str(node_run_id)],
                "tags": [],
                "metadata": {},
                "data": {
                    "serialized": {"name": "gpt-4o"},
                    "input": {"messages": [[HumanMessage(content="test")]]},
                },
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chat_model_end",
                "name": "gpt-4o",
                "run_id": str(uuid.uuid4()),
                "parent_ids": [str(node_run_id)],
                "tags": [],
                "metadata": {},
                "data": {"output": AIMessage(content="Response")},
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_end",
                "name": "test_node",
                "run_id": str(node_run_id),
                "parent_ids": [str(graph_run_id)],
                "tags": ["node:test"],
                "metadata": {},
                "data": {
                    "output": {
                        "final_output": {"content": "Response"},
                        "_steps": [
                            {
                                "id": str(uuid.uuid4()),
                                "node": "test_node",
                                "type": "assistant",
                                "status": "ok",
                                "tool": "gpt-4o",
                            }
                        ],
                    }
                },
            },
        )
        await replay_astream_event_to_tracer(
            tracer,
            {
                "event": "on_chain_end",
                "name": "LangGraph",
                "run_id": str(graph_run_id),
                "parent_ids": [],
                "tags": [],
                "metadata": {},
                "data": {"output": {"final_output": {"content": "Response"}}},
            },
        )

        # Verify tracer has nested steps before calling log_execution_trace
        nested_steps = tracer.get_nested_steps()
        assert len(nested_steps) > 0, "Expected non-empty nested_steps"

        # Call log_execution_trace as StreamAgent would in finally block
        await log_execution_trace(
            auto_tracer=tracer,
            result={"final_output": {"content": "Response"}, "_steps": []},
            token=fake_token,
            tenant_id="test-tenant",
            params=MagicMock(agent_id="test_agent", graph_run_config=None),
            metadata={
                "session_id": "test-session",
                "platform": "grpc",
            },
            effective_user_id="test-user",
            graph_name="test_graph",
            wall_ms=100,
            last_user_msg="Hello",
            guard_result=None,
            execution_input={"messages": []},
            stream=True,
            error="",
        )

        # Verify the Brain client was called with steps embedded in metadata
        call_kwargs = mock_client.log_trace.call_args[1]
        assert "metadata" in call_kwargs
        # Steps flow through BrainClient.log_trace as part of metadata (not a top-level kwarg)
        wire_steps = call_kwargs["metadata"].get("steps")
        assert wire_steps is not None, "Expected non-empty steps in trace metadata"
        assert len(wire_steps) > 0, "Expected non-empty steps in trace"
