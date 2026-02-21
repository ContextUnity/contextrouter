"""SecureTool — BaseTool subclass with mandatory permission enforcement.

Every tool in ContextRouter MUST be (or be wrapped by) a SecureTool.
Raw BaseTool instances are auto-wrapped by ``register_tool()``.

Security model:
  - ``_enforce_permission()`` is called before every ``_run()``/``_arun()``
  - No token → PermissionError (fail-closed)
  - Token without matching permission → PermissionError
  - Infrastructure tools skip checks via ``skip_auth`` flag (set ONLY by
    internal Router code, never by external registration or wrap())

Contract:
  - ContextCore defines WHAT to check: ``has_tool_access()``, ``ToolScope``
  - SecureTool defines HOW to enforce at runtime (LangChain integration)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


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

    # Tenant binding — if set, the tool can only be executed by tokens
    # that include this tenant in their ``allowed_tenants``.
    # Set during registration (project_id from RegisterTools payload).
    # Empty string = no tenant restriction (internal Router tools).
    bound_tenant: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def _effective_permission(self) -> str:
        """Return the permission string, defaulting to ``tool:{name}``."""
        return self.required_permission or f"tool:{self.name}"

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

        from contextrouter.cortex.runtime_context import get_current_access_token

        token = get_current_access_token()
        if token is None:
            raise PermissionError(
                f"No access token — tool '{self.name}' blocked (fail-closed). "
                f"Required permission: {self._effective_permission()}"
            )

        # ── Tenant isolation ─────────────────────────────────────────
        # If tool is bound to a tenant, verify the token is allowed.
        # admin:all bypasses tenant isolation (god-mode for dashboard).
        if self.bound_tenant:
            allowed = getattr(token, "allowed_tenants", ()) or ()
            perms = getattr(token, "permissions", ()) or ()
            if "admin:all" not in perms and self.bound_tenant not in allowed:
                raise PermissionError(
                    f"Tenant isolation: tool '{self.name}' is bound to "
                    f"tenant '{self.bound_tenant}', but token allows: "
                    f"{list(allowed)}.  Cross-project access denied."
                )

        # ── Permission check (fail-closed) ────────────────────────────
        try:
            from contextcore.permissions import has_tool_access
        except ImportError:
            raise PermissionError(
                f"contextcore.permissions unavailable — tool '{self.name}' "
                f"blocked (fail-closed). Install contextcore to enable "
                f"permission verification."
            ) from None

        if not has_tool_access(token.permissions, self.name):
            raise PermissionError(
                f"Permission denied for tool '{self.name}'. "
                f"Required: {self._effective_permission()}. "
                f"Token has: {list(token.permissions)}"
            )

    def _inject_authoritative_context(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Overwrite sensitive kwargs with authoritative values from the access token.

        This prevents LLM prompt injection from spoofing tenant_id, user_id, or permissions.
        Tools can blindly accept these arguments knowing they are cryptographically verified.
        """
        if self.skip_auth:
            return kwargs

        from contextrouter.cortex.runtime_context import get_current_access_token

        token = get_current_access_token()
        if not token:
            return kwargs

        secure_kwargs = dict(kwargs)

        # Determine valid parameters for this tool
        allowed_keys = set(secure_kwargs.keys())
        schema = getattr(self, "args_schema", getattr(self.wrapped_tool, "args_schema", None))
        if schema:
            if hasattr(schema, "model_fields"):  # Pydantic v2
                allowed_keys.update(schema.model_fields.keys())
            elif hasattr(schema, "__fields__"):  # Pydantic v1
                allowed_keys.update(schema.__fields__.keys())
        elif self.wrapped_tool and hasattr(self.wrapped_tool, "func"):
            import inspect

            sig = inspect.signature(self.wrapped_tool.func)
            allowed_keys.update(sig.parameters.keys())

        # 1. Tenant ID forgery prevention
        if "tenant_id" in allowed_keys:
            requested = secure_kwargs.get("tenant_id")
            if getattr(token, "allowed_tenants", ()):
                if not requested or not token.can_access_tenant(requested):
                    secure_kwargs["tenant_id"] = token.allowed_tenants[0]

        # 2. User ID forgery prevention
        if "user_id" in allowed_keys:
            user_id = getattr(token, "user_id", None)
            if user_id:
                secure_kwargs["user_id"] = user_id

        # 3. Permissions & Token ID
        if "permissions" in allowed_keys:
            secure_kwargs["permissions"] = list(getattr(token, "permissions", []))
        if "token_id" in allowed_keys:
            secure_kwargs["token_id"] = getattr(token, "token_id", "")

        return secure_kwargs

    def _run(
        self,
        *args: Any,
        run_manager: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous execution with permission enforcement."""
        self._enforce_permission()
        secure_kwargs = self._inject_authoritative_context(kwargs)

        if self.wrapped_tool is not None:
            import inspect

            sig = inspect.signature(self.wrapped_tool._run)
            if "run_manager" in sig.parameters or any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
            ):
                secure_kwargs["run_manager"] = run_manager
            if "config" in sig.parameters or any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
            ):
                secure_kwargs["config"] = config
            return self.wrapped_tool._run(*args, **secure_kwargs)
        raise NotImplementedError(
            f"SecureTool '{self.name}' has no _run implementation and no wrapped tool."
        )

    async def _arun(
        self,
        *args: Any,
        run_manager: Any = None,
        config: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Async execution with permission enforcement."""
        self._enforce_permission()
        secure_kwargs = self._inject_authoritative_context(kwargs)

        if self.wrapped_tool is not None:
            import inspect

            sig = inspect.signature(self.wrapped_tool._arun)
            if "run_manager" in sig.parameters or any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
            ):
                secure_kwargs["run_manager"] = run_manager
            if "config" in sig.parameters or any(
                p.kind == p.VAR_KEYWORD for p in sig.parameters.values()
            ):
                secure_kwargs["config"] = config
            return await self.wrapped_tool._arun(*args, **secure_kwargs)
        raise NotImplementedError(
            f"SecureTool '{self.name}' has no _arun implementation and no wrapped tool."
        )

    @classmethod
    def wrap(
        cls,
        tool: BaseTool,
        *,
        permission: str = "",
        scope: str = "read",
        tenant: str = "",
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
            # Already secure — update permission/tenant if provided
            if permission and not tool.required_permission:
                tool.required_permission = permission
            if tenant and not tool.bound_tenant:
                tool.bound_tenant = tenant
            return tool

        effective_permission = permission or f"tool:{tool.name}"

        init_kwargs: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description or f"Wrapped tool: {tool.name}",
            "required_permission": effective_permission,
            "required_scope": scope,
            "wrapped_tool": tool,
            "bound_tenant": tenant,
            # skip_auth intentionally NOT set — always False for wrapped tools
        }

        # Preserve args_schema if present
        if hasattr(tool, "args_schema") and tool.args_schema is not None:
            init_kwargs["args_schema"] = tool.args_schema

        return cls(**init_kwargs)

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

        secure = cls(
            name=tool.name,
            description=tool.description or f"Infra tool: {tool.name}",
            required_permission=f"infra:{tool.name}",
            wrapped_tool=tool,
            skip_auth=True,
        )
        logger.debug("Wrapped + marked tool '%s' as infra (skip_auth=True)", tool.name)
        return secure

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
