"""PlatformToolRegistry — typed dispatch for internal service tools.
Maps tool_binding strings to (executor, config_schema, required_scopes).
Used by make_platform_node() to dispatch compiled graph nodes to real
Brain/Shield/Zero/Worker gRPC calls.
Security by Construction:
- Every tool declares required token scopes at registration time
- Scope check runs BEFORE tool execution — fail-closed
- Config validated via Pydantic schema at registration time
- Unknown bindings raise PlatformServiceError
- Duplicate registrations are rejected
"""

from __future__ import annotations

from dataclasses import dataclass

from contextunity.core import ContextToken, get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError, SecurityError
from contextunity.core.types import JsonDict
from pydantic import BaseModel, ValidationError

from contextunity.router.cortex.compiler.platform_tools.helpers.contracts import PlatformExecutor

logger = get_contextunit_logger(__name__)

# Known service prefixes for routing validation
_KNOWN_PREFIXES = ("brain_", "shield_", "worker_", "language_", "router_")


@dataclass(frozen=True)
class ToolRegistration:
    """Immutable record of a registered platform tool."""

    binding: str
    executor: PlatformExecutor
    config_schema: type[BaseModel]
    required_scopes: list[str]
    service_prefix: str


class PlatformToolRegistry:
    """Registry for platform service tools.

    Thread-safe for reads after initialization. Tools are registered
    at import time (service startup), then only read during execution.
    """

    def __init__(self) -> None:
        """Initialise an empty tool registry (populated at import time)."""
        self._tools: dict[str, ToolRegistration] = {}

    def register(
        self,
        binding: str,
        executor: PlatformExecutor,
        config_schema: type[BaseModel],
        required_scopes: list[str],
    ) -> None:
        """Register a platform tool.

        Args:
            binding: Tool binding name (e.g. 'brain_search').
            executor: Async function(state, config) -> dict.
            config_schema: Pydantic model for config validation.
            required_scopes: Token scopes required to execute.

        Raises:
            PlatformServiceError: If binding already registered or prefix unknown.
        """
        if binding in self._tools:
            raise PlatformServiceError(
                message=f"Platform tool '{binding}' is already registered.",
                tool_binding=binding,
            )

        # Extract service prefix
        service_prefix = None
        for prefix in _KNOWN_PREFIXES:
            if binding.startswith(prefix):
                service_prefix = prefix.rstrip("_")
                break

        if service_prefix is None:
            raise PlatformServiceError(
                message=(
                    f"Unknown service prefix in tool binding '{binding}'. "
                    f"Expected prefixes: {sorted(_KNOWN_PREFIXES)}"
                ),
                tool_binding=binding,
            )

        self._tools[binding] = ToolRegistration(
            binding=binding,
            executor=executor,
            config_schema=config_schema,
            required_scopes=required_scopes,
            service_prefix=service_prefix,
        )

        logger.info(
            "Registered platform tool: %s (service: %s, scopes: %s)",
            binding,
            service_prefix,
            required_scopes,
        )

    def get(self, binding: str) -> ToolRegistration:
        """Get a registered tool by binding name."""
        if binding not in self._tools:
            raise PlatformServiceError(
                message=(
                    f"Platform tool '{binding}' is not registered. "
                    f"Available tools: {sorted(self._tools.keys())}"
                ),
                tool_binding=binding,
            )
        return self._tools[binding]

    def list_bindings(self) -> list[str]:
        """Return all registered tool binding names."""
        return sorted(self._tools.keys())

    def validate_config(self, binding: str, config_dict: JsonDict) -> BaseModel:
        """Validate *config_dict* against the Pydantic schema registered for *binding*."""
        registration = self.get(binding)
        try:
            return registration.config_schema(**config_dict)
        except ValidationError as e:
            raise PlatformServiceError(
                message=f"Platform tool '{binding}' config validation failed",
                tool_binding=binding,
            ) from e

    def check_scopes(self, binding: str, token: ContextToken | None) -> None:
        """Verify token has required scopes for this tool."""
        registration = self.get(binding)

        if token is None:
            raise SecurityError(
                message=(f"Platform tool '{binding}' requires a valid token. No token provided."),
                tool_binding=binding,
            )

        token_perms = list(token.permissions)

        for scope in registration.required_scopes:
            # ``has_permission`` uses pre-expanded ``_effective_permissions``
            # (inheritance/wildcards). Literal ``scope in token_perms`` alone was
            # the pre-d9bb324 behaviour; the fallback was added in the compiler
            # extract so ``brain:write`` tokens satisfy ``brain:read`` tools.
            if scope in token_perms or token.has_permission(scope):
                continue
            raise SecurityError(
                message=(
                    f"Platform tool '{binding}' requires scope '{scope}'. Token has: {token_perms}"
                ),
                tool_binding=binding,
                required_scope=scope,
            )

    def has(self, binding: str) -> bool:
        """Check if a binding is registered."""
        return binding in self._tools


# Module-level singleton
platform_registry = PlatformToolRegistry()

__all__ = ["PlatformToolRegistry", "ToolRegistration", "platform_registry"]
