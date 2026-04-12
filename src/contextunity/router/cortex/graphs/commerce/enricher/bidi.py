"""
Enricher BiDi tools client.

Provides type-safe tools for cu.commerce manipulation during enrichment.
"""

from __future__ import annotations

from typing import Any, Dict

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)


class EnricherBiDi:
    """Commerce BiDi client for product enrichment."""

    def __init__(self, trace_id: str, tenant_id: str = "traverse") -> None:
        from contextunity.router.service.stream_executors import get_stream_executor_manager

        self._manager = get_stream_executor_manager()
        self.trace_id = trace_id
        self.tenant_id = tenant_id

    async def verify_technologies(self, names: list[str]) -> list[str]:
        """Verify which technologies exist. Returns list of MISSING ones."""
        res = await self._manager.execute(
            self.tenant_id,
            "verify_technologies",
            {"names": names},
            timeout=30,
        )
        return res.get("missing", []) if isinstance(res, dict) else []

    async def create_wagtail_technology(self, payload: Dict[str, Any]) -> int:
        """Create a new technology snippet/page in Wagtail."""
        res = await self._manager.execute(
            self.tenant_id,
            "create_wagtail_technology",
            payload,
            timeout=60,
        )
        return res.get("id", 0) if isinstance(res, dict) else 0

    async def save_enriched_product(self, product_id: int, enrichment: Dict[str, Any]) -> bool:
        """Update a product with full enrichment data."""
        res = await self._manager.execute(
            self.tenant_id,
            "save_enriched_product",
            {"product_id": product_id, "enrichment": enrichment},
            timeout=60,
        )
        return res.get("success", False) if isinstance(res, dict) else False
