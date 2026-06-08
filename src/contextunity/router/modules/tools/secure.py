"""SecureTool — BaseTool subclass with mandatory permission enforcement.

Every tool in contextunity.router MUST be (or be wrapped by) a SecureTool.
Raw BaseTool instances are auto-wrapped by ``register_tool()``.

Security model:
  - ``_enforce_permission()`` is called before every ``_run()``/``_arun()``
  - No token → PermissionError (fail-closed)
  - Token without matching permission → PermissionError
  - Infrastructure tools skip checks via ``skip_auth`` flag (set ONLY by
    internal Router code, never by external registration or wrap())

Contract:
  - contextunity.router and contextunity.core define WHAT to check: ``has_tool_access()``, ``ToolScope``
  - SecureTool defines HOW to enforce at runtime (LangChain integration)
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from contextvars import Token
from typing import ClassVar, TypeGuard

from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.exceptions import SecurityError
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import ConfigDict
from typing_extensions import override

from contextunity.router.langchain_boundaries import tool_arun_method, tool_run_method

logger = get_contextunit_logger(__name__)


def _mapping_str_keys(fields: object) -> frozenset[str]:
    if is_object_dict(fields):
        return frozenset(fields)
    return frozenset()


def _schema_field_names(schema: object) -> frozenset[str]:
    if isinstance(schema, type) and hasattr(schema, "model_fields"):
        model_fields_obj: object = getattr(schema, "model_fields", None)
        return _mapping_str_keys(model_fields_obj)
    return _mapping_str_keys(getattr(schema, "model_fields", None)) or _mapping_str_keys(
        getattr(schema, "__fields__", None)
    )


def _tag_strings(raw: object) -> list[str]:
    if not is_object_list(raw):
        return []
    tags: list[str] = []
    for element in raw:
        tags.append(str(element))
    return tags


def _callable_param_names(func: object) -> frozenset[str]:
    if not callable(func):
        return frozenset()
    return frozenset(inspect.signature(func).parameters.keys())


def _is_awaitable_object(value: object) -> TypeGuard[Awaitable[object]]:
    return inspect.isawaitable(value)


class SecureTool(BaseTool):
    """BaseTool with mandatory permission enforcement.

    Every tool call is guarded by ``_enforce_permission()`` which checks
    the current ``ContextToken`` from runtime context.

    Attributes:
        required_permission: Permission string needed to invoke this tool.
            Defaults to ``tool:{name}`` if not explicitly set.
        required_scope: ToolScope for this tool's primary operation
            (read / write / admin).

    Security:
        The ``skip_auth`` flag exempts infrastructure tools from permission
        checks.  It defaults to ``False`` and can ONLY be set via the
        ``mark_infra()`` classmethod — never through ``wrap()`` or
        ``register_tool()``.  This prevents malicious tools from naming
        themselves ``"log_execution_trace"`` to bypass enforcement.

    Usage::

        # Explicit: tool with declared permission
        class MyTool(SecureTool):
            name = "my_tool"
            description = "Does something"
            required_permission = "tool:my_tool"
            required_scope = "read"

            def _run(self, query: str) -> str:
                self._enforce_permission()
                return do_something(query)

        # Auto-wrap: any raw BaseTool
        raw_tool = some_langchain_tool()
        secure = SecureTool.wrap(raw_tool, permission="tool:some_tool")

        # Infrastructure tool (internal Router code only):
        secure = SecureTool.mark_infra(trace_tool)
    """

    required_permission: str = ""
    required_scope: str = "read"

    # The original tool being wrapped (None if SecureTool was subclassed directly).
    wrapped_tool: BaseTool | None = None

    # If True, skip permission checks.
    # Defaults to False. ONLY set via mark_infra() — never via wrap() or
    # register_tool().  This prevents name-spoofing attacks where an
    # external tool names itself "log_execution_trace" to bypass auth.
    skip_auth: bool = False

    # Tenant binding — tokens must intersect this set to execute the tool.
    bound_allowed_tenants: tuple[str, ...] = ()
    # Legacy single-tenant binding; prefer bound_allowed_tenants.
    bound_tenant: str = ""

    model_config: ClassVar[ConfigDict] = ConfigDict(arbitrary_types_allowed=True)

    def _effective_permission(self) -> str:
        """Return the permission string, defaulting to ``tool:{name}``."""
        return self.required_permission or f"tool:{self.name}"

    def _effective_bound_tenants(self) -> tuple[str, ...]:
        """Return tool-bound tenant scope (multi-tenant preferred over legacy single)."""
        if self.bound_allowed_tenants:
            return self.bound_allowed_tenants
        if self.bound_tenant:
            return (self.bound_tenant,)
        return ()

    def _resolve_execution_tenants(self, token: ContextToken) -> tuple[str, ...] | None:
        """Intersect token scope with tool binding; None means no tenant attenuation."""
        bound = self._effective_bound_tenants()
        token_tenants = token.allowed_tenants
        if not bound:
            return None
        if not token_tenants:
            return bound
        intersected = tuple(tenant for tenant in bound if tenant in set(token_tenants))
        return intersected

    def _enforce_permission(self) -> None:
        """Check access token permissions AND tenant binding before execution.

        Fail-closed: if no token is present OR the token lacks the required
        permission, raises PermissionError.

        Tenant check: if ``bound_tenant`` is set, verifies that the token's
        ``allowed_tenants`` includes this tenant.  Prevents cross-project
        tool access (e.g. project A's user invoking project B's SQL tool).

        Tools marked with ``skip_auth=True`` (via ``mark_infra()``) skip
        all checks.
        """
        if self.skip_auth:
            return

        from contextunity.router.core.context import get_current_access_token

        token = get_current_access_token()
        if token is None:
            raise SecurityError(
                (
                    f"No access token — tool '{self.name}' blocked (fail-closed). "
                    f"Required permission: {self._effective_permission()}"
                )
            )

        # ── Tenant isolation ─────────────────────────────────────────
        allowed_scope = self._effective_bound_tenants()
        if allowed_scope:
            allowed = getattr(token, "allowed_tenants", ()) or ()
            perms = getattr(token, "permissions", ()) or ()
            if "admin:all" not in perms and not set(allowed_scope).intersection(allowed):
                raise SecurityError(
                    (
                        f"Tenant isolation: tool '{self.name}' is bound to tenants "
                        f"{list(allowed_scope)}, but token allows: {list(allowed)}. "
                        "Cross-project access denied."
                    )
                )

        # ── Permission check (fail-closed) ────────────────────────────
        from contextunity.core.authz import authorize

        effective_perm = self._effective_permission()
        use_tool_name = effective_perm.startswith("tool:")

        decision = authorize(
            token,
            tool_name=self.name if use_tool_name else None,
            permission=effective_perm,
            service="router",
            tool_scope=self.required_scope or None,
        )
        if decision.denied:
            raise SecurityError(
                (
                    f"Permission denied for tool '{self.name}'. "
                    f"{decision.reason}. "
                    f"Required: {effective_perm}"
                )
            )

    def _inject_authoritative_context(self, kwargs: dict[str, object]) -> dict[str, object]:
        """Overwrite sensitive kwargs with authoritative values from the access token.

        This prevents LLM prompt injection from spoofing tenant_id, user_id, or permissions.
        Tools can blindly accept these arguments knowing they are cryptographically verified.
        """
        if self.skip_auth:
            return kwargs

        from contextunity.router.core.context import get_current_access_token

        token = get_current_access_token()
        if not token:
            return kwargs

        secure_kwargs: dict[str, object] = dict(kwargs)

        allowed_keys = set(secure_kwargs.keys())
        schema_obj: object | None = getattr(self, "args_schema", None)
        if schema_obj is None and self.wrapped_tool is not None:
            schema_obj = getattr(self.wrapped_tool, "args_schema", None)
        if schema_obj is not None:
            allowed_keys.update(_schema_field_names(schema_obj))
        elif self.wrapped_tool is not None:
            func_obj: object | None = getattr(self.wrapped_tool, "func", None)
            if func_obj is not None:
                allowed_keys.update(_callable_param_names(func_obj))

        if "tenant_id" in allowed_keys:
            allowed = token.allowed_tenants
            requested_raw = secure_kwargs.get("tenant_id")
            if allowed:
                allowed_set = set(allowed)
                if isinstance(requested_raw, str) and requested_raw in allowed_set:
                    secure_kwargs["tenant_id"] = requested_raw
                else:
                    secure_kwargs["tenant_id"] = sorted(allowed)[0]
            else:
                _ = secure_kwargs.pop("tenant_id", None)

        if "user_id" in allowed_keys:
            user_id = getattr(token, "user_id", None)
            if user_id:
                secure_kwargs["user_id"] = user_id

        if "permissions" in allowed_keys:
            secure_kwargs["permissions"] = list(getattr(token, "permissions", []))
        if "token_id" in allowed_keys:
            secure_kwargs["token_id"] = getattr(token, "token_id", "")

        return secure_kwargs

    def _resolve_provenance_prefix(self) -> str:
        """Determine if this tool should show as 'federated_tool' in provenance.

        This is purely an observability concern — it does NOT affect permissions.
        Permissions always use the canonical 'tool:' prefix.
        """
        my_tags = _tag_strings(getattr(self, "tags", None))

        wrapped_tags: list[str] = []
        if self.wrapped_tool is not None:
            wrapped_tags = _tag_strings(getattr(self.wrapped_tool, "tags", None))

        if "federated" in my_tags or "federated" in wrapped_tags:
            return "federated_tool"

        eff_perm = self._effective_permission()
        if eff_perm.startswith("privacy:"):
            return "privacy_tool"
        if eff_perm.startswith("shield:"):
            return "shield_tool"

        return "tool"

    def _prepare_execution(
        self, kwargs: dict[str, object]
    ) -> tuple[dict[str, object], Token[ContextToken | None] | None]:
        """Shared provenance + attenuation logic for _run/_arun.

        Returns:
            (secure_kwargs, token_ref) — token_ref is None if no attenuation occurred.
        """
        from contextunity.core.tokens import TokenBuilder

        from contextunity.router.core.context import (
            append_provenance,
            get_current_access_token,
            set_current_access_token,
        )

        self._enforce_permission()
        secure_kwargs = self._inject_authoritative_context(kwargs)

        token = get_current_access_token()
        token_ref: Token[ContextToken | None] | None = None

        prefix = self._resolve_provenance_prefix()

        mode = getattr(self, "required_scope", "execute")
        if token and not self.skip_auth:
            if getattr(token, "permissions", None):
                for perm in token.permissions:
                    if perm.startswith(f"tool:{self.name}:"):
                        mode = perm.split(":")[-1]
                        break

            try:
                narrowed_tenants = self._resolve_execution_tenants(token)
                agent_id = f"{prefix}:{self.name}:{mode}"
                if narrowed_tenants is not None:
                    attenuated = TokenBuilder().attenuate(
                        token,
                        permissions=None,
                        allowed_tenants=narrowed_tenants,
                        agent_id=agent_id,
                    )
                else:
                    attenuated = TokenBuilder().attenuate(
                        token,
                        permissions=None,
                        agent_id=agent_id,
                    )
                token_ref = set_current_access_token(attenuated)
            except (SecurityError, ValueError, TypeError, AttributeError) as e:
                # Fail-closed: if attenuation cannot be produced we must not run
                # the tool with the un-attenuated parent token (capability-strip
                # would be bypassed). Refuse execution and surface the error.
                raise SecurityError(
                    (
                        f"Token attenuation failed for tool '{self.name}': {e}. "
                        "Refusing to execute with un-attenuated parent token."
                    )
                ) from e

        if prefix != "privacy_tool":
            append_provenance(f"{prefix}:{self.name}:{mode}")

        return secure_kwargs, token_ref

    def _run(
        self,
        *args: object,
        run_manager: CallbackManagerForToolRun | None = None,
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> object:
        """Synchronous execution with permission enforcement."""
        from contextunity.router.core.context import reset_current_access_token

        tool_kwargs: dict[str, object] = dict(kwargs)
        secure_kwargs, token_ref = self._prepare_execution(tool_kwargs)

        try:
            if self.wrapped_tool is not None:
                run_method = tool_run_method(self.wrapped_tool)
                sig = inspect.signature(run_method)
                if "run_manager" in sig.parameters or any(
                    p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    secure_kwargs["run_manager"] = run_manager
                if "config" in sig.parameters or any(
                    p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    secure_kwargs["config"] = config
                return run_method(*args, **secure_kwargs)
            raise NotImplementedError(
                f"SecureTool '{self.name}' has no _run implementation and no wrapped tool."
            )
        finally:
            if token_ref is not None:
                reset_current_access_token(token_ref)

    async def _arun(
        self,
        *args: object,
        run_manager: CallbackManagerForToolRun | None = None,
        config: RunnableConfig | None = None,
        **kwargs: object,
    ) -> object:
        """Async execution with permission enforcement."""
        from contextunity.router.core.context import reset_current_access_token

        tool_kwargs: dict[str, object] = dict(kwargs)
        secure_kwargs, token_ref = self._prepare_execution(tool_kwargs)

        try:
            if self.wrapped_tool is not None:
                arun_method = tool_arun_method(self.wrapped_tool)
                sig = inspect.signature(arun_method)
                if "run_manager" in sig.parameters or any(
                    p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    secure_kwargs["run_manager"] = run_manager
                if "config" in sig.parameters or any(
                    p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
                ):
                    secure_kwargs["config"] = config
                pending: object = arun_method(*args, **secure_kwargs)
                if _is_awaitable_object(pending):
                    return await pending
                return pending
            raise NotImplementedError(
                f"SecureTool '{self.name}' has no _arun implementation and no wrapped tool."
            )
        finally:
            if token_ref is not None:
                reset_current_access_token(token_ref)

    @classmethod
    def wrap(
        cls,
        tool: BaseTool,
        *,
        permission: str = "",
        scope: str = "read",
        tenant: str = "",
        allowed_tenants: tuple[str, ...] = (),
    ) -> SecureTool:
        """Wrap a raw BaseTool in SecureTool with permission enforcement.

        The original tool's ``_run``/``_arun`` are delegated to by the wrapper.
        Name, description, and args_schema are preserved.

        **Never sets ``skip_auth``** — wrapped tools always require a token.

        Args:
            tool: The original BaseTool to wrap.
            permission: Required permission (default: ``tool:{name}``).
            scope: Required scope (default: ``read``).
            tenant: Bind tool to a specific tenant/project.  If set,
                only tokens with this tenant in ``allowed_tenants``
                can execute the tool.

        Returns:
            SecureTool instance wrapping the original.
        """
        if isinstance(tool, SecureTool):
            if permission and not tool.required_permission:
                tool.required_permission = permission
            if allowed_tenants and not tool.bound_allowed_tenants:
                tool.bound_allowed_tenants = allowed_tenants
            elif tenant and not tool.bound_tenant:
                tool.bound_tenant = tenant
            return tool

        effective_permission = permission or f"tool:{tool.name}"
        tool_schema = getattr(tool, "args_schema", None)
        tool_description = getattr(tool, "description", None)

        if tool_schema is not None:
            return cls(
                name=tool.name,
                description=tool_description or f"Wrapped tool: {tool.name}",
                args_schema=tool_schema,
                required_permission=effective_permission,
                required_scope=scope,
                wrapped_tool=tool,
                bound_allowed_tenants=allowed_tenants,
                bound_tenant=tenant,
            )
        return cls(
            name=tool.name,
            description=tool_description or f"Wrapped tool: {tool.name}",
            required_permission=effective_permission,
            required_scope=scope,
            wrapped_tool=tool,
            bound_allowed_tenants=allowed_tenants,
            bound_tenant=tenant,
        )

    @classmethod
    def mark_infra(cls, tool: BaseTool) -> SecureTool:
        """Mark a tool as infrastructure — exempt from permission checks.

        Use ONLY for internal Router mechanics (trace logging, PII cleanup).
        Do NOT use for user-facing tools.

        This is the ONLY way to set ``skip_auth=True``.

        Args:
            tool: Tool to mark as infrastructure.

        Returns:
            SecureTool with ``skip_auth=True``.
        """
        if isinstance(tool, SecureTool):
            tool.skip_auth = True
            logger.debug("Marked tool '%s' as infra (skip_auth=True)", tool.name)
            return tool

        tool_description = getattr(tool, "description", None)
        secure = cls(
            name=tool.name,
            description=tool_description or f"Infra tool: {tool.name}",
            required_permission=f"infra:{tool.name}",
            wrapped_tool=tool,
            skip_auth=True,
        )
        logger.debug("Wrapped + marked tool '%s' as infra (skip_auth=True)", tool.name)
        return secure

    @override
    def __repr__(self) -> str:
        wrapped_info = f", wrapped={self.wrapped_tool.name}" if self.wrapped_tool else ""
        skip_info = ", infra=True" if self.skip_auth else ""
        tenant_info = f", tenant={self.bound_tenant!r}" if self.bound_tenant else ""
        return (
            f"SecureTool(name={self.name!r}, "
            f"permission={self._effective_permission()!r}, "
            f"scope={self.required_scope!r}"
            f"{wrapped_info}{skip_info}{tenant_info})"
        )


__all__ = ["SecureTool"]
