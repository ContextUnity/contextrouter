"""Security configuration for ContextRouter."""

from pydantic import BaseModel, ConfigDict, Field

from .base import DEFAULT_READ_PERMISSION, DEFAULT_WRITE_PERMISSION


class SecurityPoliciesConfig(BaseModel):
    """Security policies for data access control."""

    model_config = ConfigDict(extra="ignore")

    read_permission: str = DEFAULT_READ_PERMISSION
    write_permission: str = DEFAULT_WRITE_PERMISSION


class SecurityConfig(BaseModel):
    """Security settings for the application."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False  # Disabled by default; enable in production
    policies: SecurityPoliciesConfig = Field(default_factory=SecurityPoliciesConfig)

    # Basic token settings
    token_ttl_seconds: int = 3600  # 1 hour
    token_issuer: str = "contextrouter"

    # Biscuit token settings
    private_key_path: str = ""  # Path to private key for token signing
