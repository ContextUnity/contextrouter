"""Langfuse telemetry for LangGraph/LangChain.

Per-project isolation via request-level credentials
----------------------------------------------------
Each request can carry its own Langfuse settings in the execution metadata:

    metadata = {
        "langfuse_enabled": True,           # False → skip tracing entirely
        "langfuse_project_id": "proj-xxx",  # Project ID for dashboard URL
        "langfuse_secret_key": "sk-...",    # optional; falls back to router global
        "langfuse_public_key": "pk-...",    # optional; falls back to router global
        "langfuse_host": "https://...",     # optional; falls back to router global
    }

These are passed from the project's client code (e.g. RouterClient.execute_agent)
via the request payload.  The router reads them before creating any traces.

Client cache
------------
Langfuse clients are cached by (host, public_key) so a per-project client is
created once and reused for subsequent requests from the same project.
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from contextcore import get_context_unit_logger

from contextrouter.core import get_core_config

logger = get_context_unit_logger(__name__)

_warned_missing_langfuse = False
_global_initialized = False
_global_client: object | None = None

# Cache: (host, public_key) → Langfuse client | None
_client_cache: dict[tuple[str, str], object | None] = {}


# ---------------------------------------------------------------------------
# Per-request Langfuse context — passed down from request metadata
# ---------------------------------------------------------------------------


@dataclass
class LangfuseRequestCtx:
    """Langfuse settings extracted from request metadata.

    This is the clean alternative to env-var-per-tenant hacks.
    The project provides these values in the request payload metadata.
    """

    enabled: bool = False
    secret_key: str = ""  # project-specific; falls back to global if empty
    public_key: str = ""  # project-specific; falls back to global if empty
    project_id: str = ""  # project-specific; falls back to global if empty
    host: str = ""  # project-specific; falls back to global if empty

    @classmethod
    def from_metadata(cls, metadata: dict | None) -> "LangfuseRequestCtx":
        """Extract Langfuse settings from request execution metadata."""
        if not metadata:
            return cls()
        enabled_raw = metadata.get("langfuse_enabled", True)
        if isinstance(enabled_raw, str):
            enabled = enabled_raw.lower() not in ("0", "false", "no", "off")
        else:
            enabled = bool(enabled_raw)
        return cls(
            enabled=enabled,
            secret_key=metadata.get("langfuse_secret_key", "") or "",
            public_key=metadata.get("langfuse_public_key", "") or "",
            project_id=metadata.get("langfuse_project_id", "") or "",
            host=metadata.get("langfuse_host", "") or "",
        )

    def effective_secret_key(self) -> str:
        return self.secret_key or get_core_config().langfuse.secret_key

    def effective_public_key(self) -> str:
        return self.public_key or get_core_config().langfuse.public_key

    def effective_host(self) -> str:
        return self.host or get_core_config().langfuse.host

    def effective_project_id(self) -> str:
        return self.project_id or get_core_config().langfuse.project_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _langfuse_available() -> bool:
    global _warned_missing_langfuse
    if importlib.util.find_spec("langfuse") is None:
        if not _warned_missing_langfuse:
            _warned_missing_langfuse = True
            logger.warning(
                "Langfuse keys are set but the `langfuse` package is not installed. "
                "Install with `pip install contextrouter[observability]` to enable tracing."
            )
        return False
    return True


def _enabled(ctx: LangfuseRequestCtx | None = None) -> bool:
    """Return True if tracing is enabled for this request context."""
    if ctx is not None and not ctx.enabled:
        return False
    if ctx is not None:
        has_creds = bool(ctx.effective_secret_key() and ctx.effective_public_key())
    else:
        cfg = get_core_config()
        has_creds = bool(cfg.langfuse.secret_key and cfg.langfuse.public_key)
    if not has_creds:
        return False
    return _langfuse_available()


def _ensure_threading_instrumented() -> None:
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


def _get_or_create_client(ctx: LangfuseRequestCtx | None) -> object | None:
    """Return the Langfuse client for the given request context.

    Uses the global client when the project doesn't supply its own credentials.
    Caches per (host, public_key) pair.
    """
    global _global_initialized, _global_client

    if ctx is None:
        _ctx = LangfuseRequestCtx()
    else:
        _ctx = ctx

    secret_key = _ctx.effective_secret_key()
    public_key = _ctx.effective_public_key()
    host = _ctx.effective_host()

    if not secret_key or not public_key:
        return None
    if not _langfuse_available():
        return None

    cache_key = (host, public_key)

    if cache_key in _client_cache:
        return _client_cache[cache_key]

    # Check if this is the global client (same creds as router config)
    cfg = get_core_config()
    is_global = (
        secret_key == cfg.langfuse.secret_key
        and public_key == cfg.langfuse.public_key
        and host == cfg.langfuse.host
    )

    if is_global and _global_initialized:
        _client_cache[cache_key] = _global_client
        return _global_client

    # Need to instrument threading exactly once
    if not _global_initialized:
        import os

        service_name = cfg.langfuse.service_name
        if service_name:
            os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
        _ensure_threading_instrumented()

    try:
        from langfuse import Langfuse  # type: ignore[import-not-found]

        lf = Langfuse(secret_key=secret_key, public_key=public_key, host=host)
        if not lf.auth_check():
            logger.warning(
                "Langfuse auth_check failed for public_key=%s; traces may not export",
                public_key[:12] + "...",
            )
        else:
            logger.info("Langfuse client ready (host=%s pk=%s...)", host, public_key[:12])

        _client_cache[cache_key] = lf

        if is_global:
            _global_client = lf
            _global_initialized = True

        return lf
    except Exception:
        logger.exception("Langfuse client initialization failed")
        _client_cache[cache_key] = None
        if is_global:
            _global_initialized = True
        return None


def _log_error_cleanly(context: str, exc: Exception) -> None:
    logger.error("%s failed: %s: %s", context, type(exc).__name__, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_langfuse_callbacks(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    platform: str | None = None,
    tags: list[str] | None = None,
    langfuse_ctx: LangfuseRequestCtx | None = None,
) -> list[object]:
    """Return LangChain callback handlers for Langfuse tracing.

    Pass ``langfuse_ctx`` built from the request metadata to apply
    project-specific settings (enabled flag, credentials, project_id).
    """
    if not _enabled(langfuse_ctx):
        return []

    _ = session_id, user_id, platform, tags
    try:
        lf = _get_or_create_client(langfuse_ctx)
        if lf is None:
            return []
        try:
            from langfuse.langchain import CallbackHandler  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            logger.info("Langfuse callback handler disabled (optional dependency missing): %s", exc)
            return []
        try:
            handler = CallbackHandler(client=lf)
        except TypeError:
            handler = CallbackHandler()
        return [handler]
    except Exception:
        logger.exception("Failed to create Langfuse callback handler")
        return []


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
    langfuse_ctx: LangfuseRequestCtx | None = None,
) -> Generator[object | None, None, None]:
    if not _enabled(langfuse_ctx):
        yield None
        return

    lf = _get_or_create_client(langfuse_ctx)
    if lf is None:
        yield None
        return

    try:
        from langfuse import propagate_attributes  # type: ignore[import-not-found]
    except Exception as e:
        _log_error_cleanly("trace_context initialization", e)
        yield None
        return

    context = None
    if trace_id:
        hex_id = trace_id.replace("-", "")
        if len(hex_id) == 32:
            context = {"trace_id": hex_id}
            if parent_span_id:
                context["parent_span_id"] = parent_span_id
        else:
            logger.warning("Invalid trace_id for Langfuse (must be 32 hex chars): %s", trace_id)

    cfg = get_core_config()
    environment = cfg.langfuse.environment

    if trace_tags is None:
        tags: list[str] = []
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
        with lf.start_as_current_observation(  # type: ignore[union-attr]
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
        if not span_initialized:
            _log_error_cleanly(f"trace_context({name}) initialization", e)
            yield None
        raise


@contextmanager
def retrieval_span(
    *,
    name: str,
    input_data: dict[str, object] | None = None,
    langfuse_ctx: LangfuseRequestCtx | None = None,
) -> Generator[dict[str, object], None, None]:
    """Create a Langfuse span for a retrieval-like operation."""
    ctx: dict[str, object] = {"output": None, "metadata": None}
    if not _enabled(langfuse_ctx):
        yield ctx
        return

    lf = _get_or_create_client(langfuse_ctx)
    if lf is None:
        yield ctx
        return

    span_initialized = False
    try:
        with lf.start_as_current_observation(as_type="span", name=name) as span:  # type: ignore[union-attr]
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


def flush(langfuse_ctx: LangfuseRequestCtx | None = None) -> None:
    """Flush pending Langfuse events."""
    if not _enabled(langfuse_ctx):
        return
    lf = _get_or_create_client(langfuse_ctx)
    if lf is None:
        return
    try:
        lf.flush()  # type: ignore[union-attr]
    except Exception:
        logger.exception("Langfuse flush failed")


def get_langfuse_trace_id() -> str:
    """Extract the current Langfuse/OTel trace ID (32 hex chars)."""
    ctx = get_current_trace_context()
    if ctx:
        return ctx.get("trace_id", "")
    return ""


def get_langfuse_trace_url(langfuse_ctx: LangfuseRequestCtx | None = None) -> str:
    """Build a direct Langfuse dashboard URL for the current active trace."""
    trace_id = get_langfuse_trace_id()
    if not trace_id:
        return ""
    if langfuse_ctx:
        host = langfuse_ctx.effective_host().rstrip("/")
        project_id = langfuse_ctx.effective_project_id()
    else:
        cfg = get_core_config()
        host = cfg.langfuse.host.rstrip("/")
        project_id = cfg.langfuse.project_id
    if project_id:
        return f"{host}/project/{project_id}/traces?peek={trace_id}"
    return f"{host}/traces?peek={trace_id}"


__all__ = [
    "LangfuseRequestCtx",
    "get_langfuse_callbacks",
    "trace_context",
    "retrieval_span",
    "get_current_trace_context",
    "get_langfuse_trace_id",
    "get_langfuse_trace_url",
    "flush",
]
