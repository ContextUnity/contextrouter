"""Router admin platform tools — Brain Admin RPC observability (admin:read gated).

Tenant resolution follows ``redis_memory._resolve_tenant``: the effective tenant
comes from the verified caller token, never from the agent-supplied parameter alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import SecurityError
from contextunity.core.types import ContextUnitPayload, JsonDict

from contextunity.router.langchain_boundaries import tool

if TYPE_CHECKING:
    from contextunity.core.sdk import BrainClient
from contextunity.router.modules.tools import register_tool

logger = get_contextunit_logger(__name__)


def _require_admin_read() -> None:
    from contextunity.router.modules.tools.auth_context import resolve_tool_context_token

    token = resolve_tool_context_token()
    if not token.has_permission("admin:read"):
        raise SecurityError("Admin tool: admin:read permission required")


def _resolve_tenant(requested: str | None) -> str | None:
    """Derive effective tenant from the verified caller token (fail-closed)."""
    from contextunity.router.modules.tools.auth_context import resolve_tool_context_token

    token = resolve_tool_context_token()
    if token.has_permission("admin:all"):
        return requested
    allowed = token.allowed_tenants
    if not allowed:
        raise SecurityError("Admin tool: token grants no tenant access")
    if requested is None:
        if len(allowed) == 1:
            return allowed[0]
        raise SecurityError(
            "Admin tool: tenant_id is required when the token allows multiple tenants"
        )
    if requested not in allowed:
        raise SecurityError(f"Admin tool: tenant '{requested}' is not allowed by the caller token")
    return requested


def _get_brain_client() -> BrainClient:
    from contextunity.core.sdk import BrainClient

    from contextunity.router.core import get_core_config
    from contextunity.router.modules.tools.auth_context import resolve_tool_context_token

    config = get_core_config()
    return BrainClient(host=config.brain_url, token=resolve_tool_context_token())


@tool
async def query_traces(
    tenant_id: str | None = None,
    agent_id: str | None = None,
    hours: int | None = 24,
    limit: int = 50,
    offset: int = 0,
) -> ContextUnitPayload:
    """Search agent traces via Brain AdminSearchTraces (admin:read required)."""
    _require_admin_read()
    effective_tenant = _resolve_tenant(tenant_id)
    brain = _get_brain_client()
    return await brain.admin_search_traces(
        tenant_id=effective_tenant,
        agent_id=agent_id,
        hours=hours,
        limit=limit,
        offset=offset,
    )


@tool
async def get_analytics_summary(
    tenant_id: str | None = None,
    hours: int | None = 24,
) -> JsonDict:
    """Rich analytics summary via Brain AdminGetAnalyticsSummary (admin:read required)."""
    _require_admin_read()
    effective_tenant = _resolve_tenant(tenant_id)
    brain = _get_brain_client()
    return await brain.get_analytics_summary(
        tenant_id=effective_tenant,
        hours=hours,
    )


@tool
async def list_platform_tenants() -> list[JsonDict]:
    """List tenants visible to the admin token via Brain ListTenants (admin:read required)."""
    _require_admin_read()
    brain = _get_brain_client()
    return await brain.list_tenants()


_ADMIN_TOOLS = [
    query_traces,
    get_analytics_summary,
    list_platform_tenants,
]

for _t in _ADMIN_TOOLS:
    register_tool(_t, permission="admin:read")

logger.info("Registered %d Brain admin platform tools", len(_ADMIN_TOOLS))

__all__ = [
    "get_analytics_summary",
    "list_platform_tenants",
    "query_traces",
]
