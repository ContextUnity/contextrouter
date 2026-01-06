"""Access control + token primitives (Biscuit-style).

Per architecture corrections:
- keep token primitives in `core/` (kernel responsibility)
- provide optional enforcement via AccessManager (config-driven)

NOTE: legacy `contextrouter.security.token_builder` has been removed (no shims).
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from contextrouter.core import BisquitEnvelope, Config, get_core_config


@dataclass(frozen=True)
class BiscuitToken:
    """Minimal token representation used by framework interfaces."""

    token_id: str
    permissions: tuple[str, ...] = ()
    exp_unix: float | None = None

    def is_expired(self, *, now: float | None = None) -> bool:
        if self.exp_unix is None:
            return False
        t = time.time() if now is None else now
        return t >= self.exp_unix


class TokenBuilder:
    """Token minting + attenuation + verification.

    If/when `biscuit-auth` is adopted, this becomes a thin wrapper around it.
    """

    def __init__(self, *, enabled: bool, private_key_path: str | None = None) -> None:
        self._enabled = enabled
        self._private_key_path = private_key_path

    @property
    def enabled(self) -> bool:
        return self._enabled

    def mint_root(
        self, *, user_ctx: dict[str, Any], permissions: Iterable[str], ttl_s: float
    ) -> BiscuitToken:
        _ = user_ctx  # reserved for future datalog facts
        token_id = secrets.token_urlsafe(16)
        exp_unix = time.time() + float(ttl_s)
        return BiscuitToken(token_id=token_id, permissions=tuple(permissions), exp_unix=exp_unix)

    def attenuate(
        self,
        token: BiscuitToken,
        *,
        permissions: Iterable[str] | None = None,
        ttl_s: float | None = None,
    ) -> BiscuitToken:
        exp_unix = token.exp_unix
        if ttl_s is not None:
            exp_unix = min(exp_unix or (time.time() + ttl_s), time.time() + ttl_s)
        perms = token.permissions if permissions is None else tuple(permissions)
        return BiscuitToken(token_id=token.token_id, permissions=perms, exp_unix=exp_unix)

    def verify(self, token: BiscuitToken, *, required_permission: str) -> None:
        if not self._enabled:
            return
        if not isinstance(token, BiscuitToken):
            raise PermissionError("Missing token")
        if token.is_expired():
            raise PermissionError("Token expired")
        if required_permission not in token.permissions:
            raise PermissionError(f"Missing permission: {required_permission}")


def require_permission(permission: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for enforcing permissions at provider boundaries.

    Additionally, when an envelope is present, ensure `envelope.token_id` matches
    the token for audit-trail consistency (Bisquit protocol).
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            token = kwargs.get("token")
            if not isinstance(token, BiscuitToken):
                raise PermissionError("Missing token")
            if permission not in token.permissions:
                raise PermissionError(f"Missing permission: {permission}")

            env = kwargs.get("envelope") or kwargs.get("data")
            if isinstance(env, BisquitEnvelope):
                if env.token_id is None:
                    env.sign(token.token_id)
                if env.token_id != token.token_id:
                    raise PermissionError("Envelope token_id does not match token")

            return await fn(*args, **kwargs)

        return wrapper

    return decorator


@dataclass(frozen=True)
class AccessManager:
    """Authorization gate for providers/sinks."""

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

    def verify_read(self, token: BiscuitToken, *, permission: str | None = None) -> None:
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

    def verify_write(self, token: BiscuitToken, *, permission: str | None = None) -> None:
        """Verify write permission (back-compat accepts `permission=` kwarg)."""
        required = (
            str(permission).strip()
            if isinstance(permission, str) and str(permission).strip()
            else self.config.security.policies.write_permission
        )
        self.token_builder.verify(token, required_permission=required)

    def verify_envelope_write(self, envelope: BisquitEnvelope, token: BiscuitToken) -> None:
        """Verify write permission and ensure envelope.token_id matches the token id.

        Principal spec: Providers must verify the `token_id` on the Bisquit envelope
        for write operations.
        """

        self.verify_write(token)

        # If security is disabled, do not enforce token_id presence/match.
        if not self.config.security.enabled:
            return

        env_token_id = envelope.token_id
        tok_token_id = token.token_id

        # If the token has an id, ensure the envelope carries it for audit trails.
        if tok_token_id and env_token_id is None:
            envelope.sign(tok_token_id)
            env_token_id = tok_token_id

        if not env_token_id:
            raise PermissionError(
                "write denied: BisquitEnvelope.token_id is required when security is enabled"
            )
        if tok_token_id and env_token_id != tok_token_id:
            raise PermissionError(
                "write denied: BisquitEnvelope.token_id does not match the provided token"
            )


__all__ = ["BiscuitToken", "TokenBuilder", "require_permission", "AccessManager"]
