"""Platform tool infrastructure — generic executor factory and config injection.

Each platform tool file contains only business logic + a Pydantic config class.
The executor wrapping (config injection, error handling, PlatformServiceError)
is handled ONCE here via ``make_platform_executor``.
"""

from __future__ import annotations

import inspect

from contextunity.core import ContextToken
from contextunity.core.exceptions import PlatformServiceError, SecurityError
from contextunity.core.types import is_json_dict, is_object_dict
from pydantic import BaseModel

from contextunity.router.cortex.types import GraphState, StateUpdate, is_graph_state

from .contracts import PlatformExecutor, PlatformResult, PlatformToolFunc


def _non_null_config_fields(config: BaseModel) -> dict[str, object]:
    """Extract non-null, non-dunder config fields from a Pydantic model dump."""
    dumped = config.model_dump(exclude_none=True)
    if not is_object_dict(dumped):
        return {}
    return {key: value for key, value in dumped.items() if not key.startswith("__")}


def resolve_tenant_from_state(state: GraphState, *, binding: str = "") -> str:
    """Derive tenant_id from ``state['__token__']`` — token is SPOT.

    Platform tools must NEVER trust ``state['tenant_id']`` (or any
    other state-provided identity field): those are client-writable
    and bypass tenant isolation. The only authoritative source is the
    signed ``ContextToken`` placed into ``state['__token__']`` by the
    graph compiler's platform node executor.

    Raises:
        SecurityError: If no token is present or it carries no
            ``allowed_tenants`` — fail-closed, no ``"default"`` fallback.
    """
    token = state.get("__token__")
    if token is None:
        raise SecurityError(
            message=(
                f"Platform tool{f' {binding!r}' if binding else ''} requires a valid "
                "token. No token in state."
            ),
            tool_binding=binding,
        )
    allowed = token.allowed_tenants
    if not allowed:
        raise SecurityError(
            message=(
                f"Token has no allowed_tenants; cannot resolve tenant for "
                f"tool{f' {binding!r}' if binding else ''}."
            ),
            tool_binding=binding,
        )
    return allowed[0]


def resolve_tenant_and_token(state: GraphState, *, binding: str = "") -> tuple[str, ContextToken]:
    """Derive (tenant_id, token) from state — convenience for executors.

    Calls ``resolve_tenant_from_state`` for tenant isolation, then returns
    the validated token alongside the tenant_id so callers don't need a
    separate ``state.get("__token__")``.
    """
    tenant_id = resolve_tenant_from_state(state, binding=binding)
    # Token existence guaranteed by resolve_tenant_from_state (raises if None)
    token = state.get("__token__")
    if not isinstance(token, ContextToken):
        raise SecurityError(
            message=f"Platform tool{f' {binding!r}' if binding else ''} requires a valid token.",
            tool_binding=binding,
        )
    return tenant_id, token


def inject_config_into_state(state: GraphState, config: BaseModel) -> StateUpdate:
    """Create a shallow copy of state with config values injected.

    Config fields are placed inside state["__manifest_node_config__"]
    so that existing node functions can resolve them via
    ``get_node_manifest_config()``.

    Defense-in-depth: ``__``-prefixed keys are skipped to prevent
    collision with platform-internal state keys (``__token__``, etc.).
    """
    merged: StateUpdate = dict(state)
    config_dict = _non_null_config_fields(config)
    if config_dict:
        raw_config = merged.get("__manifest_node_config__")
        existing: dict[str, object] = dict(raw_config) if is_json_dict(raw_config) else {}
        existing.update(config_dict)
        merged["__manifest_node_config__"] = existing
    return merged


def inject_sql_config_into_state(state: GraphState, config: BaseModel) -> StateUpdate:
    """Inject Pydantic config fields directly into state for SQL node consumption.

    Unlike RAG tools which use ``__manifest_node_config__``, SQL nodes
    read config from top-level state keys.

    Defense-in-depth: rejects ``__``-prefixed keys to prevent dunder collision.

    Returns ``StateUpdate`` because config injection adds arbitrary keys
    that are not part of ``GraphState``.
    """
    from contextunity.core import get_contextunit_logger

    logger = get_contextunit_logger(__name__)
    merged: StateUpdate = dict(state)
    for key, value in _non_null_config_fields(config).items():
        if key.startswith("__"):
            logger.warning("Skipping __-prefixed config key: %s", key)
            continue
        merged[key] = value
    return merged


def make_platform_executor(
    func: PlatformToolFunc,
    binding: str,
    *,
    inject: str = "rag",
) -> PlatformExecutor:
    """Create a platform tool executor that wraps a tool function.

    This is the SINGLE place where config injection + error wrapping happen.
    Individual tool files contain only logic — no PlatformServiceError,
    no try/except, no registration boilerplate.

    Args:
        func: The tool function to wrap.  Can be sync or async.
              Must accept ``(state: dict) -> dict``.
        binding: Tool binding name (for error messages).
        inject: RouterConfig injection mode — ``"rag"`` uses
                ``__manifest_node_config__``, ``"sql"`` uses top-level
                state keys.
    """
    use_sql_inject = inject == "sql"

    async def _executor(state: GraphState, config: BaseModel) -> PlatformResult:
        try:
            if use_sql_inject:
                merged_update = inject_sql_config_into_state(state, config)
            else:
                merged_update = inject_config_into_state(state, config)
            if not is_graph_state(merged_update):
                raise PlatformServiceError(
                    message=f"{binding} config injection produced invalid graph state",
                    tool_binding=binding,
                )
            merged_state = merged_update
            result: object = func(merged_state)
            if inspect.isawaitable(result):
                raw = await result
            else:
                raw = result
            if not is_object_dict(raw):
                raise PlatformServiceError(
                    message=f"{binding} returned non-object result",
                    tool_binding=binding,
                )
            return {str(key): val for key, val in raw.items()}
        except PlatformServiceError:
            raise
        except Exception as exc:  # wraps-to-domain: re-raises as typed exception
            raise PlatformServiceError(
                message=f"{binding} execution failed",
                tool_binding=binding,
            ) from exc

    _executor.__qualname__ = f"platform_executor[{binding}]"
    return _executor


__all__ = [
    "inject_config_into_state",
    "inject_sql_config_into_state",
    "make_platform_executor",
    "resolve_tenant_and_token",
    "resolve_tenant_from_state",
]
