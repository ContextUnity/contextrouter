"""BiDi Client for RLM bulk matcher communication with cu.commerce."""

from __future__ import annotations

from typing import Any, Dict, List

from contextunity.core import get_contextunit_logger

logger = get_contextunit_logger(__name__)


class BiDiClient:
    """Wrapper for BiDi streaming execution to manage RLM bulk tasks."""

    def __init__(self, run_id: str):
        from contextunity.router.service.stream_executors import get_stream_executor_manager

        self._manager = get_stream_executor_manager()
        self.run_id = run_id

    async def report(self, **kwargs: Any) -> None:
        """Fire-and-forget progress report via BiDi."""
        try:
            await self._manager.execute(
                "traverse",
                "report_matcher_progress",
                {"run_id": self.run_id, **kwargs},
                timeout=5,
            )
        except Exception as e:
            logger.debug("Failed to report progress: %s", e)

    async def export_taxonomies(self) -> Dict[str, List[str]]:
        """Fetch and flatten taxonomies."""
        res = await self._manager.execute("traverse", "export_taxonomies", {}, timeout=30)
        raw = res if isinstance(res, dict) else {}
        taxonomies = {}
        for key, items in raw.items():
            if isinstance(items, list):
                taxonomies[key] = [
                    item.get("name", "") if isinstance(item, dict) else str(item)
                    for item in items
                    if (item.get("name") if isinstance(item, dict) else item)
                ]
        return taxonomies

    async def export_manual_matches(self) -> tuple[List[Dict], List[Dict]]:
        """Fetch manual manual matches and wrong pairs."""
        res = await self._manager.execute(
            "traverse", "export_manual_matches", {"limit": 1000}, timeout=30
        )
        return res.get("pairs", []), res.get("wrong_pairs", [])

    async def export_matcher_brands(self, dealer_code: str) -> List[Dict]:
        """Fetch matchable Oscar brands."""
        res = await self._manager.execute(
            "traverse",
            "export_matcher_brands",
            {"dealer": dealer_code},
            timeout=30,
        )
        return res.get("brands", [])

    async def check_abort(self) -> bool:
        """Check if matching has been aborted via UI."""
        try:
            res = await self._manager.execute(
                "traverse", "check_matcher_abort", {"run_id": self.run_id}, timeout=3
            )
            return bool(res and isinstance(res, dict) and res.get("aborted"))
        except Exception:
            return False

    async def export_unmatched_products(
        self, dealer_code: str, brand: str, force_not_matched: bool
    ) -> List[Dict]:
        """Fetch unmatched supplier products for a brand."""
        res = await self._manager.execute(
            "traverse",
            "export_unmatched_products",
            {
                "dealer": dealer_code,
                "brand": brand,
                "limit": 50000,
                "force_not_matched": force_not_matched,
            },
            timeout=60,
        )
        return res.get("products", [])

    async def export_site_products(self, brand: str) -> List[Dict]:
        """Fetch site products for a brand."""
        res = await self._manager.execute(
            "traverse",
            "export_site_products",
            {"brand": brand, "limit": 50000},
            timeout=60,
        )
        return res.get("products", [])

    async def bulk_link_products(self, matches: List[Dict]) -> int:
        """Upload bulk matches via BiDi."""
        save_res = await self._manager.execute(
            "traverse",
            "bulk_link_products",
            {"matches": matches, "mode": "candidate"},
            timeout=60,
        )
        return save_res.get("linked", 0)
