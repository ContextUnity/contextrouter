"""Langfuse telemetry for LangGraph/LangChain."""

from __future__ import annotations

import importlib.util
import logging
from contextlib import contextmanager
from typing import Generator

from contextrouter.core import get_core_config

logger = logging.getLogger(__name__)

_initialized = False
_warned_missing_langfuse = False


def _enabled() -> bool:
    cfg = get_core_config()
    enabled_by_keys = bool(cfg.langfuse.public_key and cfg.langfuse.secret_key)
    if not enabled_by_keys:
        return False

    # Langfuse is an optional dependency (installed via `contextrouter[observability]`).
    # If users set keys but didn't install the extra, degrade gracefully.
    if importlib.util.find_spec("langfuse") is None:
        global _warned_missing_langfuse
        if not _warned_missing_langfuse:
            _warned_missing_langfuse = True
            logger.warning(
                "Langfuse keys are set but the `langfuse` package is not installed. "
                "Install with `pip install contextrouter[observability]` to enable tracing."
            )
        return False

    return True


def _get_environment() -> str:
    return get_core_config().langfuse.environment


def _get_service_name() -> str:
    return get_core_config().langfuse.service_name


def _ensure_initialized() -> None:
    global _initialized
    if _initialized or not _enabled():
        return

    try:
        try:
            from opentelemetry.instrumentation.threading import (  # type: ignore[import-not-found]
                ThreadingInstrumentor,
            )

            ThreadingInstrumentor().instrument()
            logger.debug("ThreadingInstrumentor enabled for context propagation")
        except ImportError:
            logger.debug("opentelemetry-instrumentation-threading not available")
        except Exception as e:
            logger.warning("Failed to instrument threading for tracing: %s", e)

        from langfuse import get_client  # type: ignore[import-not-found]

        lf = get_client()
        if not lf.auth_check():
            logger.warning("Langfuse auth_check failed; traces may not be exported")
        else:
            logger.info("Langfuse initialized successfully")
        _initialized = True
    except Exception:
        logger.exception("Langfuse initialization failed")


def get_langfuse_callbacks(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    platform: str | None = None,
    tags: list[str] | None = None,
) -> list[object]:
    if not _enabled():
        return []

    _ = session_id, user_id, platform, tags
    try:
        _ensure_initialized()
        try:
            from langfuse.langchain import (  # type: ignore[import-not-found]
                CallbackHandler,  # type: ignore[import-not-found]
            )
        except ModuleNotFoundError as exc:
            logger.info(
                "Langfuse callback handler disabled (optional dependency missing): %s",
                exc,
            )
            return []

        handler = CallbackHandler()
        return [handler]
    except Exception:
        logger.exception("Failed to create Langfuse callback handler")
        return []


def _log_error_cleanly(context: str, exc: Exception) -> None:
    error_type = type(exc).__name__
    error_msg = str(exc)
    logger.error("%s failed: %s: %s", context, error_type, error_msg)


def get_current_trace_context() -> dict[str, str] | None:
    try:
        from opentelemetry import trace as otel_trace  # type: ignore[import-not-found]

        span = otel_trace.get_current_span()
        span_ctx = span.get_span_context() if span is not None else None
        if span_ctx is None or not getattr(span_ctx, "is_valid", False):
            return None
        trace_id = format(span_ctx.trace_id, "032x")
        parent_span_id = format(span_ctx.span_id, "016x")
        return {"trace_id": trace_id, "parent_span_id": parent_span_id}
    except Exception:
        logger.exception("Failed to get current OTel span context")
        return None


@contextmanager
def trace_context(
    *,
    session_id: str,
    platform: str,
    name: str = "rag_request",
    user_id: str | None = None,
    trace_input: object | None = None,
    trace_metadata: dict[str, object] | None = None,
    trace_tags: list[str] | None = None,
    trace_id: str | None = None,
    parent_span_id: str | None = None,
    tenant_id: str | None = None,
    agent_id: str | None = None,
    graph_name: str | None = None,
) -> Generator[object | None, None, None]:
    if not _enabled():
        yield None
        return

    try:
        _ensure_initialized()
        from langfuse import get_client, propagate_attributes  # type: ignore[import-not-found]

        lf = get_client()
    except Exception as e:
        _log_error_cleanly("trace_context initialization", e)
        yield None
        return

    # Prepare OTel context for Langfuse if trace_id is provided
    # Langfuse uses OTel context to link observations.
    context = None
    if trace_id:
        # Langfuse/OTel trace_id MUST be 32-char hex.
        # If we got a UUID with dashes, strip them.
        hex_id = trace_id.replace("-", "")
        if len(hex_id) == 32:
            context = {"trace_id": hex_id}
            if parent_span_id:
                context["parent_span_id"] = parent_span_id
        else:
            logger.warning("Invalid trace_id for Langfuse (must be 32 hex chars): %s", trace_id)

    environment = _get_environment()
    service_name = _get_service_name()
    _ = service_name

    # Build tags: tenant, agent, graph (filterable in Langfuse UI)
    if trace_tags is None:
        tags = []
        if tenant_id:
            tags.append(f"tenant:{tenant_id}")
        if agent_id:
            tags.append(f"agent:{agent_id}")
        if graph_name:
            tags.append(f"graph:{graph_name}")
        if not tags:
            tags = [platform]
    else:
        tags = list(trace_tags)

    span_initialized = False
    try:
        with lf.start_as_current_observation(
            as_type="span", name=name, trace_context=context
        ) as span:
            span_initialized = True
            try:
                with propagate_attributes(
                    session_id=session_id,
                    user_id=user_id,
                    tags=tags,
                    metadata={
                        "platform": platform,
                        "environment": environment,
                        "tenant_id": tenant_id or "",
                        "agent_id": agent_id or "",
                        "graph_name": graph_name or "",
                    },
                ):
                    if isinstance(trace_metadata, dict) and trace_metadata:
                        try:
                            span.update(metadata=trace_metadata)  # type: ignore[attr-defined]
                        except Exception:
                            logger.exception("Failed to set trace metadata on span")

                    if trace_input is not None:
                        try:
                            span.update(input=trace_input)  # type: ignore[attr-defined]
                        except Exception:
                            logger.exception("Failed to set trace input")

                    yield span
            except GeneratorExit:
                raise
            except Exception as e:
                _log_error_cleanly(f"trace_context({name})", e)
                raise
    except GeneratorExit:
        raise
    except Exception as e:
        # If we failed before yielding, degrade gracefully.
        if not span_initialized:
            _log_error_cleanly(f"trace_context({name}) initialization", e)
            yield None
        raise


@contextmanager
def retrieval_span(
    *, name: str, input_data: dict[str, object] | None = None
) -> Generator[dict[str, object], None, None]:
    """Create a Langfuse span for a retrieval-like operation.

    Callers may mutate the yielded dict (commonly `ctx["output"] = {...}`).
    """
    # Callers may set:
    # - ctx["output"]: JSON-serializable span output
    # - ctx["metadata"]: JSON-serializable span metadata
    ctx: dict[str, object] = {"output": None, "metadata": None}
    if not _enabled():
        yield ctx
        return

    span_initialized = False
    try:
        _ensure_initialized()
        from langfuse import get_client  # type: ignore[import-not-found]

        lf = get_client()
        with lf.start_as_current_observation(as_type="span", name=name) as span:
            span_initialized = True
            try:
                span.update(input=input_data or {})  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("Failed to update span with input data: %s", e)

            try:
                yield ctx
            except GeneratorExit:
                raise
            except Exception as e:
                _log_error_cleanly(f"retrieval_span({name})", e)
                raise

            try:
                span.update(output=ctx.get("output"))  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("Failed to update span with output data: %s", e)
            try:
                if ctx.get("metadata") is not None:
                    span.update(metadata=ctx.get("metadata"))  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("Failed to update span with metadata: %s", e)
    except GeneratorExit:
        raise
    except Exception as e:
        if not span_initialized:
            _log_error_cleanly(f"retrieval_span({name}) initialization", e)
            yield ctx
        raise


def flush() -> None:
    """Flush pending Langfuse events."""
    if not _enabled():
        return
    try:
        from langfuse import get_client  # type: ignore[import-not-found]

        get_client().flush()
    except Exception:
        logger.exception("Langfuse flush failed")


def get_langfuse_trace_id() -> str:
    """Extract the current Langfuse/OTel trace ID (32 hex chars).

    Call this inside a ``trace_context()`` block to get the active trace ID.
    Returns empty string if no active trace or Langfuse is disabled.
    """
    ctx = get_current_trace_context()
    if ctx:
        return ctx.get("trace_id", "")
    return ""


def get_langfuse_trace_url() -> str:
    """Build a direct Langfuse dashboard URL for the current active trace.

    Call inside a ``trace_context()`` block.  Returns empty string if
    Langfuse is disabled or no active trace exists.
    """
    trace_id = get_langfuse_trace_id()
    if not trace_id:
        return ""
    cfg = get_core_config()
    host = cfg.langfuse.host.rstrip("/")
    project_id = cfg.langfuse.project_id
    if project_id:
        return f"{host}/project/{project_id}/traces?peek={trace_id}"
    return f"{host}/traces?peek={trace_id}"


__all__ = [
    "get_langfuse_callbacks",
    "trace_context",
    "retrieval_span",
    "get_current_trace_context",
    "get_langfuse_trace_id",
    "get_langfuse_trace_url",
    "flush",
]
