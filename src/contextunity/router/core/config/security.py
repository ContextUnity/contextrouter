"""Security configuration for contextunity.router."""

from contextunity.core.permissions import Permissions
from pydantic import BaseModel, ConfigDict, Field


class SecurityPoliciesConfig(BaseModel):
    """Security policies for data access control.

    Uses canonical Permissions.* constants from contextunity.core.
    """

    model_config = ConfigDict(extra="ignore")

    read_permission: str = Permissions.ROUTER_EXECUTE
    write_permission: str = Permissions.ROUTER_EXECUTE

    # Default permissions granted to new tokens
    default_permissions: tuple[str, ...] = (
        Permissions.BRAIN_READ,
        Permissions.MEMORY_READ,
        Permissions.MEMORY_WRITE,
        Permissions.TRACE_READ,
        Permissions.GRAPH_RAG,
        Permissions.TOOL_BRAIN_SEARCH,
        Permissions.TOOL_WEB_SEARCH,
        Permissions.TOOL_MEMORY,
    )


class SecurityConfig(BaseModel):
    """Security settings for contextunity.router.

    Security is always enforced — there is no toggle.
    Token signing/verification is handled by contextunity.core.signing backends
    (auto-detected: HmacBackend or SessionTokenBackend).
    """

    model_config = ConfigDict(extra="ignore")

    policies: SecurityPoliciesConfig = Field(default_factory=SecurityPoliciesConfig)
