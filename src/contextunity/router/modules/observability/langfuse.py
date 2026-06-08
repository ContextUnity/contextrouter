"""Langfuse telemetry for LangGraph/LangChain.

Requests may control whether tracing is enabled, subject to project policy.
Langfuse credentials, project IDs, and hosts must come from trusted Router
configuration or an internally constructed request context; request metadata
is never used for those values.

Langfuse clients are cached by host, public key, and a secret-key fingerprint.
"""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

from contextunity.core import get_contextunit_logger
from contextunity.core.sdk.payload import get_json_dict
from langchain_core.callbacks.base import BaseCallbackHandler

from contextunity.router.core import get_core_config

logger = get_contextunit_logger(__name__)

_warned_missing_langfuse = False
_global_initialized = False
_global_client: _LangfuseClient | None = None

# Cache: (host, public_key, secret fingerprint) → Langfuse client | None
_client_cache: dict[tuple[str, str, str], _LangfuseClient | None] = {}


# ---------------------------------------------------------------------------
# Per-request Langfuse context
# ---------------------------------------------------------------------------


@dataclass
class LangfuseRequestCtx:
    """Trusted Langfuse settings plus request-controlled enablement."""

    enabled: bool = False
    secret_key: str = ""  # trusted internal override; falls back to global if empty
    public_key: str = ""  # trusted internal override; falls back to global if empty
    project_id: str = ""  # trusted internal override; falls back to global if empty
    host: str = ""  # trusted internal override; falls back to global if empty

    @classmethod
    def from_metadata(cls, metadata: dict[str, object] | None) -> "LangfuseRequestCtx":
        """Extract langfuse settings from request execution metadata."""
        if not metadata:
            return cls()

        # Extract project config from metadata to check policy
        project_config = get_json_dict(metadata, "project_config")

        # Respect manifest policy if present (tracing_enabled baseline)
        policy_tracing: bool | None = None
        policy = get_json_dict(project_config, "policy")
        langfuse_policy = get_json_dict(policy, "langfuse")
        tracing_raw = langfuse_policy.get("tracing_enabled")
        if isinstance(tracing_raw, bool):
            policy_tracing = tracing_raw
        elif isinstance(tracing_raw, str):
            policy_tracing = tracing_raw.lower() not in ("0", "false", "no", "off")

        # Request-level metadata overrides project manifest policy
        # If neither is set, fallback to default (True)
        enabled_raw = metadata.get("langfuse_enabled")
        if enabled_raw is None:
            enabled_raw = policy_tracing if policy_tracing is not None else True

        if isinstance(enabled_raw, str):
            enabled = enabled_raw.lower() not in ("0", "false", "no", "off")
        else:
            enabled = bool(enabled_raw)

        return cls(enabled=enabled)

    def effective_secret_key(self) -> str:
        """Effective secret key."""
        return self.secret_key or get_core_config().langfuse.secret_key

    def effective_public_key(self) -> str:
        """Return the instance public key, falling back to the shared config."""
        return self.public_key or get_core_config().langfuse.public_key

    def effective_host(self) -> str:
        """Return the Langfuse host URL, falling back to the shared config."""
        return self.host or get_core_config().langfuse.host

    def effective_project_id(self) -> str:
        """Return the Langfuse project ID, falling back to the shared config."""
        return self.project_id or get_core_config().langfuse.project_id


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _langfuse_available() -> bool:
    """Check whether the ``langfuse`` package is installed."""
    global _warned_missing_langfuse
    if importlib.util.find_spec("langfuse") is None:
        if not _warned_missing_langfuse:
            _warned_missing_langfuse = True
            logger.warning(
                (
                    "Langfuse keys are set but the `langfuse` package is not installed. "
                    "Install with `pip install contextunity.router[observability]` to enable tracing."
                )
            )
        return False
    return True


def _enabled(ctx: LangfuseRequestCtx | None = None) -> bool:
    """Return true if tracing is enabled for this request context."""
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


class _ObservationSpan(Protocol):
    def update(self, **kwargs: object) -> object:
        """Update the observation."""
        ...


class _ObservationContext(Protocol):
    def __enter__(self) -> _ObservationSpan:
        """Enter the observation span context."""
        ...

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
        """Exit the observation span context."""
        ...


class _LangfuseClient(Protocol):
    """Structural protocol for a Langfuse client instance (observation + flush API)."""

    def start_as_current_observation(
        self, *, as_type: str, name: str, trace_context: dict[str, str] | None = None
    ) -> _ObservationContext:
        """Open an observation span with the given *name* and type."""
        ...

    def flush(self) -> None:
        """Flush pending observations to the Langfuse backend."""
        ...


def _ensure_threading_instrumented() -> None:
    """ensure threading instrumented."""
    try:
        module = importlib.import_module("opentelemetry.instrumentation.threading")
        instrumentor_obj = getattr(module, "ThreadingInstrumentor", None)
        if not callable(instrumentor_obj):
            return
        instrumentor = instrumentor_obj()
        instrument = getattr(instrumentor, "instrument", None)
        if callable(instrument):
            _ = instrument()
        logger.debug("ThreadingInstrumentor enabled for context propagation")
    except ImportError:
        logger.debug("opentelemetry-instrumentation-threading not available")
    except Exception as e:
        logger.warning("Failed to instrument threading for tracing: %s", e)


def _flush_callback(fn: object) -> Callable[[], None]:
    """Wrap a dynamic Langfuse ``flush`` callable with a ``() -> None`` contract."""
    if not callable(fn):
        msg = "Langfuse flush must be callable"
        raise TypeError(msg)

    def flush() -> None:
        _ = fn()

    return flush


def _langfuse_client_from_object(obj: object) -> _LangfuseClient | None:
    """Narrow a dynamically imported Langfuse instance to the local protocol."""
    start_obs = getattr(obj, "start_as_current_observation", None)
    flush_fn = getattr(obj, "flush", None)
    auth_check = getattr(obj, "auth_check", None)
    if not callable(start_obs) or not callable(flush_fn):
        return None
    start_callable: Callable[..., object] = start_obs
    if callable(auth_check) and not bool(auth_check()):
        logger.warning("Langfuse auth_check failed; traces may not export")
    return _LangfuseClientAdapter(
        start_as_current_observation=start_callable,
        flush=_flush_callback(flush_fn),
    )


class _LangfuseClientAdapter:
    """Thin adapter so dynamic Langfuse SDK objects satisfy ``_LangfuseClient``."""

    def __init__(
        self,
        *,
        start_as_current_observation: Callable[..., object],
        flush: Callable[[], None],
    ) -> None:
        self._start_as_current_observation: Callable[..., object] = start_as_current_observation
        self._flush: Callable[[], None] = flush

    def start_as_current_observation(
        self, *, as_type: str, name: str, trace_context: dict[str, str] | None = None
    ) -> _ObservationContext:
        ctx_obj = self._start_as_current_observation(
            as_type=as_type, name=name, trace_context=trace_context
        )
        return _ObservationContextAdapter(ctx_obj)

    def flush(self) -> None:
        self._flush()


class _ObservationContextAdapter:
    """Adapter for Langfuse observation context managers."""

    def __init__(self, inner: object) -> None:
        self._inner: object = inner

    def __enter__(self) -> _ObservationSpan:
        enter = getattr(self._inner, "__enter__", None)
        if not callable(enter):
            return _ObservationSpanAdapter(None)
        return _ObservationSpanAdapter(enter())

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool | None:
        exit_fn = getattr(self._inner, "__exit__", None)
        if not callable(exit_fn):
            return None
        result = exit_fn(exc_type, exc, tb)
        return result if isinstance(result, bool) else None


class _ObservationSpanAdapter:
    """Adapter for Langfuse observation span update API."""

    def __init__(self, inner: object | None) -> None:
        self._inner: object | None = inner

    def update(self, **kwargs: object) -> object:
        if self._inner is None:
            return None
        update_fn = getattr(self._inner, "update", None)
        if not callable(update_fn):
            return None
        return update_fn(**kwargs)


def _get_or_create_client(ctx: LangfuseRequestCtx | None) -> _LangfuseClient | None:
    """Return the langfuse client for the given request context."""
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

    secret_fingerprint = hashlib.sha256(secret_key.encode()).hexdigest()
    cache_key = (host, public_key, secret_fingerprint)

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
            _ = os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
        _ensure_threading_instrumented()

    try:
        langfuse_module = importlib.import_module("langfuse")
        factory = getattr(langfuse_module, "Langfuse", None)
        if not callable(factory):
            raise AttributeError("Langfuse class missing from langfuse module")
        raw_client = factory(secret_key=secret_key, public_key=public_key, host=host)
        lf = _langfuse_client_from_object(raw_client)
        if lf is None:
            raise AttributeError("Langfuse client missing required methods")
        logger.info("Langfuse client ready (host=%s pk=%s...)", host, public_key[:12])

        _client_cache[cache_key] = lf

        if is_global:
            _global_client = lf
            _global_initialized = True

        return lf
    except (ImportError, AttributeError):
        logger.warning("Langfuse SDK not available — tracing disabled")
    except Exception:
        logger.exception("Langfuse client initialization failed")
        _client_cache[cache_key] = None
        if is_global:
            _global_initialized = True
        return None


def _log_error_cleanly(context: str, exc: Exception) -> None:
    """Log an error cleanly without propagating."""
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
) -> list["BaseCallbackHandler"]:
    """Return LangChain callback handlers for Langfuse tracing."""
    if not _enabled(langfuse_ctx):
        return []

    _ = session_id, user_id, platform, tags

    lf = _get_or_create_client(langfuse_ctx)
    if lf is None:
        return []

    try:
        langchain_module = importlib.import_module("langfuse.langchain")
        handler_factory = getattr(langchain_module, "CallbackHandler", None)
        if not callable(handler_factory):
            return []
    except (ImportError, AttributeError) as exc:
        logger.info(
            "Langfuse callback disabled (optional dependency missing): %s", type(exc).__name__
        )
        return []

    try:
        handler = handler_factory(client=lf)
    except TypeError:
        # Older langfuse versions don't accept client= kwarg
        handler = handler_factory()

    if not isinstance(handler, BaseCallbackHandler):
        return []
    return [handler]


def get_current_trace_context() -> dict[str, str] | None:
    """Retrieve the requested current trace context."""
    try:
        otel_trace = importlib.import_module("opentelemetry.trace")

        get_current_span = getattr(otel_trace, "get_current_span", None)
        if not callable(get_current_span):
            return None
        span = get_current_span()
        get_span_context = getattr(span, "get_span_context", None)
        span_ctx = get_span_context() if callable(get_span_context) else None
        if span_ctx is None or not bool(getattr(span_ctx, "is_valid", False)):
            return None
        trace_id_value = getattr(span_ctx, "trace_id", 0)
        span_id_value = getattr(span_ctx, "span_id", 0)
        trace_id = format(trace_id_value, "032x")
        parent_span_id = format(span_id_value, "016x")
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
    """Open a Langfuse trace span for the current request; yield the trace object and flush on exit."""
    if not _enabled(langfuse_ctx):
        yield None
        return

    lf = _get_or_create_client(langfuse_ctx)
    if lf is None:
        yield None
        return

    try:
        langfuse_module = importlib.import_module("langfuse")
        propagate_attributes_obj = getattr(langfuse_module, "propagate_attributes", None)
        if not callable(propagate_attributes_obj):
            raise AttributeError("propagate_attributes missing from langfuse module")
    except Exception as e:
        _log_error_cleanly("trace_context initialization", e)
        yield None
        return

    context: dict[str, str] | None = None
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
        with lf.start_as_current_observation(
            as_type="span", name=name, trace_context=context
        ) as span:
            span_initialized = True
            try:
                propagate_ctx = propagate_attributes_obj(
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
                )
                with _ObservationContextAdapter(propagate_ctx):
                    if isinstance(trace_metadata, dict) and trace_metadata:
                        try:
                            _ = span.update(metadata=trace_metadata)
                        except Exception:
                            logger.exception("Failed to set trace metadata on span")

                    if trace_input is not None:
                        try:
                            _ = span.update(input=trace_input)
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
        with lf.start_as_current_observation(as_type="span", name=name) as span:
            span_initialized = True
            try:
                _ = span.update(input=input_data or {})
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
                _ = span.update(output=ctx.get("output"))
            except Exception as e:
                logger.debug("Failed to update span with output data: %s", e)
            try:
                if ctx.get("metadata") is not None:
                    _ = span.update(metadata=ctx.get("metadata"))
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
    """Flush pending langfuse events."""
    if not _enabled(langfuse_ctx):
        return
    lf = _get_or_create_client(langfuse_ctx)
    if lf is None:
        return
    try:
        _ = lf.flush()
    except Exception:
        logger.exception("Langfuse flush failed")


def get_langfuse_trace_id() -> str:
    """Extract the current Langfuse/OTel trace ID."""
    ctx = get_current_trace_context()
    if ctx:
        return ctx.get("trace_id", "")
    return ""


def get_langfuse_trace_url(langfuse_ctx: LangfuseRequestCtx | None = None) -> str:
    """Build a direct langfuse dashboard url for the current active trace."""
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
