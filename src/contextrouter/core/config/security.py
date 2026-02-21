"""Security configuration for ContextRouter."""

from contextcore.permissions import Permissions
from pydantic import BaseModel, ConfigDict, Field


class SecurityPoliciesConfig(BaseModel):
    """Security policies for data access control.

    Uses canonical Permissions.* constants from contextcore.
    """

    model_config = ConfigDict(extra="ignore")

    read_permission: str = Permissions.BRAIN_READ
    write_permission: str = Permissions.BRAIN_WRITE

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
    """Security settings for the application."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False  # Disabled by default; enable in production
    environment: str = "development"  # "development", "staging", "production"
    policies: SecurityPoliciesConfig = Field(default_factory=SecurityPoliciesConfig)

    # Basic token settings
    token_ttl_seconds: int = 3600  # 1 hour
    token_issuer: str = "contextrouter"

    # ContextUnit protocol token settings
    private_key_path: str = ""  # Path to private key for token signing
