"""Shield Platform Tools — executors for compiled graph nodes.

Registers only the Shield operations that exist in the live gRPC contract.
"""

from __future__ import annotations

from typing import ClassVar

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import PlatformServiceError
from pydantic import BaseModel, ConfigDict, Field

from .helpers.base import resolve_tenant_from_state
from .helpers.contracts import PlatformResult, PlatformState
from .helpers.registration import PlatformRegistry, ToolRegistrationSpec, register_tool_specs
from .helpers.state import get_text_from_state

logger = get_contextunit_logger(__name__)


class ShieldScanConfig(BaseModel, frozen=True):
    """Config for shield_scan tool."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    categories: list[str] = Field(default_factory=list)


# ── Executor Functions ──────────────────────────────────────────────


def _get_shield_client(tenant_id: str):
    """Get ShieldClient for a tenant."""
    from contextunity.core.sdk import ShieldClient

    from contextunity.router.core.brain_token import get_brain_service_token

    return ShieldClient(
        tenant_id=tenant_id,
        token=get_brain_service_token(allowed_tenants=(tenant_id,)),
    )


async def _shield_scan_executor(state: PlatformState, config: ShieldScanConfig) -> PlatformResult:
    """Scan content for safety violations."""
    tenant_id = resolve_tenant_from_state(state, binding="shield_scan")
    content = get_text_from_state(state, "final_output")

    client = _get_shield_client(tenant_id)
    result = await client.scan(content=str(content), categories=config.categories)
    if "error" in result:
        raise PlatformServiceError(
            message=f"Shield scan failed: {result.get('message') or result['error']}",
            service_name="shield",
            tool_binding="shield_scan",
        )

    blocked = bool(result.get("blocked", False))
    allowed = bool(result.get("allowed", not blocked))
    return {"scan_result": result, "safe": allowed and not blocked}


# ── Registration ────────────────────────────────────────────────────


def register_shield_tools(registry: PlatformRegistry) -> None:
    """Register live Shield tools into a PlatformToolRegistry."""
    register_tool_specs(
        registry,
        [
            ToolRegistrationSpec(
                binding="shield_scan",
                executor=_shield_scan_executor,
                config_schema=ShieldScanConfig,
                required_scopes=["shield:scan"],
            )
        ],
    )


__all__ = ["register_shield_tools", "ShieldScanConfig"]
