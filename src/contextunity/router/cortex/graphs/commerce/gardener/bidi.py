"""BiDi client for Gardener → Commerce communication."""

from typing import Any

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)


class GardenerBiDi:
    """BiDi client for Gardener → Commerce communication."""

    def __init__(self, run_id: str, tenant_id: str = "traverse"):
        from contextunity.router.service.stream_executors import get_stream_executor_manager

        self._manager = get_stream_executor_manager()
        self.run_id = run_id
        self.tenant_id = tenant_id

    async def export_taxonomy(self) -> dict[str, Any]:
        """Fetch taxonomy data (categories, colors, sizes) as flat lists."""
        res = await self._manager.execute(self.tenant_id, "export_taxonomies", {}, timeout=30)
        return res if isinstance(res, dict) else {}

    async def export_normalized_examples(
        self,
        brand: str,
        source: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Fetch already-normalized products as few-shot examples.

        CRITICAL: source must match — dealer examples for dealer, oscar for oscar.
        """
        try:
            res = await self._manager.execute(
                self.tenant_id,
                "export_normalized_examples",
                {"brand": brand, "source": source, "limit": limit},
                timeout=30,
            )
            return res.get("examples", [])
        except Exception as e:
            logger.warning("Failed to fetch normalized examples: %s", e)
            return []

    async def export_products_for_normalization(
        self,
        brand: str,
        source: str,
        only_new: bool = True,
        batch_size: int = 50,
        ids: list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch products needing normalization."""
        try:
            res = await self._manager.execute(
                self.tenant_id,
                "export_products_for_normalization",
                {
                    "brand": brand,
                    "source": source,
                    "only_new": only_new,
                    "limit": batch_size,
                    "ids": ids or [],
                },
                timeout=60,
            )
            return res.get("products", [])
        except Exception as e:
            logger.warning("Failed to fetch products for normalization: %s", e)
            return []

    async def update_normalized_products(
        self,
        updates: list[dict[str, Any]],
        source: str,
    ) -> int:
        """Write normalization results back to Commerce DB."""
        try:
            res = await self._manager.execute(
                self.tenant_id,
                "update_normalized_products",
                {"updates": updates, "source": source},
                timeout=60,
            )
            return res.get("updated", 0)
        except Exception as e:
            logger.error("Failed to update normalized products: %s", e)
            return 0
