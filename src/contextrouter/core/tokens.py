"""Access control + token primitives (ContextUnit protocol).

This module re-exports ContextToken from contextcore and adds service-specific
extensions (AccessManager, require_permission) that integrate with contextrouter Config.

The canonical ContextToken implementation is in contextcore.tokens.

NOTE: legacy `contextrouter.security.token_builder` has been removed (no shims).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from contextcore import ContextToken, ContextUnit, TokenBuilder

from contextrouter.core import Config, get_core_config

# Re-export from contextcore for convenience
__all__ = ["ContextToken", "TokenBuilder"]


def require_permission(permission: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for enforcing permissions at provider boundaries.

    Additionally, when a ContextUnit is present:
    - Ensures unit.payload.token_id matches token for audit-trail consistency
    - Validates token against unit.security scopes if present
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = kwargs.get("token")
            if not isinstance(token, ContextToken):
                raise PermissionError("Missing token")
            if permission not in token.permissions:
                raise PermissionError(f"Missing permission: {permission}")

            env = kwargs.get("unit") or kwargs.get("data")
            if isinstance(env, ContextUnit):
                # Ensure token_id is set for audit trails
                payload = env.payload or {}
                token_id = payload.get("token_id")
                if token_id is None:
                    if env.payload is None:
                        env.payload = {}
                    env.payload["token_id"] = token.token_id
                elif token_id != token.token_id:
                    raise PermissionError("ContextUnit.token_id does not match token")

                # Validate against security scopes if present
                if env.security.read or env.security.write:
                    cfg = get_core_config()
                    builder = TokenBuilder(
                        enabled=cfg.security.enabled,
                        private_key_path=cfg.security.private_key_path,
                    )
                    # Determine operation from function name or default to read
                    op: Literal["read", "write"] = (
                        "write" if "write" in fn.__name__ or "sink" in fn.__name__ else "read"
                    )
                    builder.verify_unit_access(token, env, operation=op)

            return await fn(*args, **kwargs)

        return wrapper

    return decorator


@dataclass(frozen=True)
class AccessManager:
    """Authorization gate for providers/sinks.

    Service-specific extension that integrates TokenBuilder with contextrouter Config.
    """

    config: Config
    token_builder: TokenBuilder

    @classmethod
    def from_core_config(cls) -> "AccessManager":
        cfg = get_core_config()
        return cls(
            config=cfg,
            token_builder=TokenBuilder(
                enabled=cfg.security.enabled,
                private_key_path=cfg.security.private_key_path,
            ),
        )

    def verify_read(self, token: ContextToken, *, permission: str | None = None) -> None:
        """Verify read permission (back-compat accepts `permission=` kwarg).

        `secured(permission=...)` historically passed an explicit permission override.
        We keep this kwarg to avoid breaking providers when security is enabled.
        """
        required = (
            str(permission).strip()
            if isinstance(permission, str) and str(permission).strip()
            else self.config.security.policies.read_permission
        )
        self.token_builder.verify(token, required_permission=required)

    def verify_write(self, token: ContextToken, *, permission: str | None = None) -> None:
        """Verify write permission (back-compat accepts `permission=` kwarg)."""
        required = (
            str(permission).strip()
            if isinstance(permission, str) and str(permission).strip()
            else self.config.security.policies.write_permission
        )
        self.token_builder.verify(token, required_permission=required)

    def verify_unit_read(self, unit: ContextUnit, token: ContextToken) -> None:
        """Verify token can read from ContextUnit based on security scopes."""
        self.verify_read(token)

        if self.config.security.enabled and (unit.security.read or unit.security.write):
            self.token_builder.verify_unit_access(token, unit, operation="read")

    def verify_unit_write(self, unit: ContextUnit, token: ContextToken) -> None:
        """Verify write permission and ensure unit.payload.token_id matches token.

        Also validates token against unit.security scopes for capability-based access control.

        Principal spec: Providers must verify the `token_id` on the ContextUnit
        for write operations and validate against security scopes.
        """
        self.verify_write(token)

        # If security is disabled, do not enforce token_id presence/match.
        if not self.config.security.enabled:
            return

        payload = unit.payload or {}
        env_token_id = payload.get("token_id")
        tok_token_id = token.token_id

        # If the token has an id, ensure the unit carries it for audit trails.
        if tok_token_id and env_token_id is None:
            if unit.payload is None:
                unit.payload = {}
            unit.payload["token_id"] = tok_token_id
            env_token_id = tok_token_id

        if not env_token_id:
            raise PermissionError(
                "write denied: ContextUnit.payload.token_id is required when security is enabled"
            )
        if tok_token_id and env_token_id != tok_token_id:
            raise PermissionError(
                "write denied: ContextUnit.payload.token_id does not match the provided token"
            )

        # Validate against security scopes
        if unit.security.read or unit.security.write:
            self.token_builder.verify_unit_access(token, unit, operation="write")


__all__ = ["ContextToken", "TokenBuilder", "require_permission", "AccessManager"]
