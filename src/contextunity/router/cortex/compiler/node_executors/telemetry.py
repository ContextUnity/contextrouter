"""Telemetry wrappers for graph node execution.
Emits timing, error classification, and tool-call telemetry events
via LangChain's ``dispatch_custom_event`` for observability backends.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, TypeVar

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import (
    ContextUnityError,
    PlatformServiceError,
)
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.callbacks import dispatch_custom_event

from contextunity.router.core.exceptions import RouterToolTimeout
from contextunity.router.cortex.config_resolution import get_node_attr
from contextunity.router.cortex.types import (
    ToolErrorData,
    ToolTelemetryPayload,
    is_registered_project_config,
)

from ...compiler.types import NodeMeta
from ...events import BrainEvent
from ...types import GraphState

if TYPE_CHECKING:
    from contextunity.core.manifest.router import RetryPolicy
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables.config import RunnableConfig

    from contextunity.router.modules.models.base import BaseLLM
    from contextunity.router.modules.models.types import ModelRequest, ModelResponse

logger = get_contextunit_logger(__name__)

T = TypeVar("T")

# ── Tool telemetry ──────────────────────────────────────────────────


async def tool_telemetry(
    node_name: str,
    tool_binding: str,
    tool_kind: str,
    node_meta: NodeMeta,
    state: GraphState,
    execute_fn: Callable[[], Awaitable[T]],
    *,
    tool_args: object = None,
) -> T:
    """Wrap tool execution with BrainEvent telemetry and error classification.

    Emits ``tool_result`` events via ``dispatch_custom_event`` containing
    timing, status, and error metadata. Classifies errors as timeout,
    platform-level, or generic execution failures.

    Args:
        node_name: Logical node identifier for trace display.
        tool_binding: Tool binding string (e.g., ``federated:sql``).
        tool_kind: Tool category (``platform``, ``federated``).
        node_meta: Optional metadata dict from the node spec.
        state: Current graph execution state.
        execute_fn: Zero-arg async callable that performs the actual work.
        tool_args: Optional args dict included in the telemetry payload.

    Returns:
        The result of ``execute_fn()``.

    Raises:
        PlatformServiceError: Wraps unhandled exceptions with context.
    """
    _ = state
    started_at = time.perf_counter()
    tool_name = node_meta.get("tool_name", tool_binding)
    handler = node_meta.get("handler", tool_name)
    source = node_meta.get("source", "registry")
    toolkit = node_meta.get("toolkit")
    args = tool_args

    def _emit(status: str, result: object, error: ToolErrorData | None) -> None:
        """Emit a ``tool_result`` BrainEvent with timing and error data."""
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        payload: ToolTelemetryPayload = {
            "status": status,
            "duration_ms": duration_ms,
            "tool_kind": tool_kind,
            "tool_binding": tool_binding,
            "handler": handler,
            "source": source,
            "toolkit": toolkit,
            "args": args,
            "result": result,
            "error": error,
        }
        try:
            dispatch_custom_event(
                "brain_event",
                {"event": BrainEvent(type="tool_result", node=node_name, data=payload)},
            )
        except RuntimeError:
            pass

    try:
        result = await execute_fn()
        _emit("ok", result, None)
        return result
    except asyncio.TimeoutError:
        _emit(
            "error",
            None,
            {
                "code": "timeout",
                "message": "Tool execution timed out",
                "retryable": True,
            },
        )
        raise
    except RouterToolTimeout:
        _emit(
            "error",
            None,
            {
                "code": "timeout",
                "message": "Federated tool timeout",
                "retryable": True,
            },
        )
        raise
    except PlatformServiceError as e:
        _emit(
            "error",
            None,
            {
                "code": "platform_service_error",
                "message": getattr(e, "message", "A platform service error occurred."),
                "retryable": False,
            },
        )
        raise
    except ContextUnityError as e:
        _emit(
            "error",
            None,
            {
                "code": getattr(e, "code", "context_unity_error"),
                "message": getattr(e, "message", "A context unity error occurred."),
                "retryable": False,
            },
        )
        raise
    except Exception as exc:  # graceful-degrade: telemetry must not crash pipeline
        logger.exception("Unhandled exception in tool %s on node %s", tool_binding, node_name)
        _emit(
            "error",
            None,
            {
                "code": "execution_error",
                "message": "An internal execution error occurred.",
                "retryable": False,
            },
        )
        raise PlatformServiceError(
            message=f"Node '{node_name}' tool '{tool_binding}' execution failed",
            node_name=node_name,
            tool_binding=tool_binding,
        ) from exc


# ── Model telemetry ─────────────────────────────────────────────────


def _resolve_llm_display_name(llm: BaseLLM, fallback_model_name: str = "") -> str:
    """Extract a human-readable model name for trace display.

    Checks ``_candidate_keys`` first (handles ``FallbackModel`` composites),
    then ``model_key``, then falls back to ``fallback_model_name`` or
    the class name.

    Args:
        llm: Model instance to extract a display name from.
        fallback_model_name: Manifest-declared model key used as last resort.

    Returns:
        A clean model name string for observability dashboards.
    """
    # model_key is always set by BaseModel.__init__; check _candidate_keys
    # first because FallbackModel.__init__ sets _model_key to a synthetic
    # "fallback/..." composite — the first candidate is the actual primary.
    candidate_keys = getattr(llm, "_candidate_keys", None)
    if is_object_list(candidate_keys) and candidate_keys:
        first_candidate = candidate_keys[0]
        if isinstance(first_candidate, str):
            return first_candidate

    model_key = getattr(llm, "model_key", None)
    if isinstance(model_key, str) and model_key and not model_key.startswith("fallback/"):
        return model_key

    return fallback_model_name or getattr(llm.__class__, "__name__", "LLM")


def _resolve_prompt_version(node_name: str, state: GraphState) -> str | None:
    """Resolve ``prompt_version`` from manifest ``project_config`` in state.

    Reads ``state["metadata"]["project_config"]`` and delegates to
    ``get_node_attr`` for per-node prompt version lookup.

    Args:
        node_name: Node to resolve version for.
        state: Graph state containing ``metadata.project_config``.

    Returns:
        Prompt version hash, or ``None`` if not configured.
    """
    metadata = state.get("metadata")
    if not is_object_dict(metadata):
        return None
    project_config_raw = metadata.get("project_config")
    if not is_registered_project_config(project_config_raw):
        return None

    val = get_node_attr(project_config_raw, node_name, "prompt_version")
    return str(val) if val is not None else None


async def model_telemetry(
    llm: BaseLLM,
    request: ModelRequest,
    config: RunnableConfig | None,
    *,
    prompt_version: str | None = None,
    node_name: str | None = None,
    state: GraphState | None = None,
    fallback_model_name: str = "",
    trace_messages: list[BaseMessage] | None = None,
    retry_policy: RetryPolicy | None = None,
) -> ModelResponse:
    """Generate with automatic LangChain callback tracing.

    This is the **single entry point** for all traced LLM calls.
    It wraps ``llm.generate()`` with proper ``on_chat_model_start``
    and ``on_llm_end`` callbacks so the observability dashboard
    displays model names, prompt versions, tokens and cost.

    Args:
        llm: The model instance (BaseLLM or FallbackModel).
        request: The ModelRequest to send.
        config: LangGraph RunnableConfig carrying callback handlers.
        prompt_version: Optional prompt version hash for provenance.
            If ``None`` and *node_name* + *state* are provided,
            prompt_version is resolved automatically from manifest
            project_config.
        node_name: Logical node name (e.g. ``"planner"``) for
            automatic prompt_version resolution.
        state: Graph execution state (must contain ``metadata``
            with ``project_config``) for automatic resolution.
        fallback_model_name: Model name from manifest (used if model
            doesn't expose a clean name, e.g. FallbackModel).
        trace_messages: Optional pre-built LangChain messages for
            trace display. If None, messages are reconstructed from
            request.system and request.parts.
        retry_policy: Retry policy forwarded to ``llm.generate()``.

    Returns:
        ModelResponse from the provider.
    """
    from langchain_core.runnables.config import RunnableConfig as _RC
    from langchain_core.runnables.config import get_async_callback_manager_for_config

    effective_config = config or _RC()
    cb_manager = get_async_callback_manager_for_config(effective_config)

    llm_name = _resolve_llm_display_name(llm, fallback_model_name)

    # Auto-resolve prompt_version from manifest if not provided explicitly
    if prompt_version is None and node_name and state is not None:
        prompt_version = _resolve_prompt_version(node_name, state)

    invocation_params = {"prompt_version": prompt_version} if prompt_version else None

    # Build LangChain-format messages for trace display
    if trace_messages is None:
        from langchain_core.messages import HumanMessage, SystemMessage

        trace_messages = []
        if request.system:
            trace_messages.append(SystemMessage(content=request.system))
        # Concatenate all text parts into a single user message
        from contextunity.router.modules.models.types import TextPart

        user_parts = [p.text for p in request.parts if isinstance(p, TextPart) and p.text]
        if user_parts:
            trace_messages.append(HumanMessage(content="\n\n".join(user_parts)))

    # ── on_chat_model_start ──────────────────────────────────────────
    rms = await cb_manager.on_chat_model_start(
        serialized={"name": llm_name},
        messages=[trace_messages],
        invocation_params=invocation_params,
    )
    run_manager = rms[0] if rms else None

    # ── llm.generate() ───────────────────────────────────────────────
    try:
        response = await llm.generate(request, retry_policy=retry_policy)
    except Exception as e:  # graceful-degrade: telemetry must not crash pipeline
        if run_manager:
            try:
                await run_manager.on_llm_error(e)
            except Exception:  # graceful-degrade: telemetry must not crash pipeline
                pass
        raise

    # ── on_llm_end ───────────────────────────────────────────────────
    if run_manager:
        try:
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, LLMResult

            gen = ChatGeneration(
                message=AIMessage(content=response.text),
                text=response.text,
            )

            llm_output: dict[str, str | dict[str, int | float]] = {}
            if response.usage:
                llm_output["token_usage"] = {
                    "prompt_tokens": response.usage.input_tokens or 0,
                    "completion_tokens": response.usage.output_tokens or 0,
                    "total_tokens": (
                        (response.usage.input_tokens or 0) + (response.usage.output_tokens or 0)
                    ),
                    "total_cost": response.usage.total_cost or 0.0,
                }

            # Attach actual model name from provider response
            if response.raw_provider:
                llm_output["model_name"] = response.raw_provider.model_name
            else:
                candidate_keys = getattr(llm, "_candidate_keys", None)
                if candidate_keys:
                    llm_output["model_name"] = candidate_keys[0]
                else:
                    llm_output["model_name"] = llm_name

            res = LLMResult(generations=[[gen]], llm_output=llm_output)
            await run_manager.on_llm_end(res)
        except Exception as cb_err:  # graceful-degrade: telemetry must not crash pipeline
            logger.warning("Failed to end LLM callback: %s", cb_err)

    return response
