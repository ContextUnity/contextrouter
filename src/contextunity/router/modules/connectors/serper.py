"""
Serper Search Connector.

High-performance connector for Serper.dev Google Search API.
Supports web, news, images search with filtering options.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import httpx
from contextunity.core import get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.narrowing import as_str
from contextunity.core.parsing import json_loads
from contextunity.core.types import JsonDict, is_json_dict, is_object_dict, is_object_list

from contextunity.router.core.config import RouterConfig

logger = get_contextunit_logger(__name__)


def _serper_result_rows(raw: object) -> list[dict[str, object]]:
    """Narrow Serper API list payloads to object dict rows."""
    if not is_object_list(raw):
        return []
    rows: list[dict[str, object]] = []
    for item in raw:
        if is_object_dict(item):
            rows.append(item)
    return rows


class SerperSearchConnector:
    """Connector for Serper.dev search API.

    Supports multiple search types:
    - Web search (general)
    - News search (returns recent articles)
    - Images search (returns URLs only)

    Usage:
        connector = SerperSearchConnector(
            query="latest AI news",
            search_type="news",
            num_results=10,
            config=config,
        )
        results = await connector.search_raw()
    """

    BASE_URL: ClassVar[str] = "https://google.serper.dev"

    def __init__(
        self,
        query: str,
        *,
        search_type: Literal["search", "news", "images"] = "search",
        num_results: int = 10,
        country: str = "us",
        language: str = "en",
        time_filter: str | None = None,
        config: RouterConfig | None = None,
    ) -> None:
        """Initialize Serper connector.

        Args:
            query: Search query string
            search_type: Type of search - "search", "news", or "images"
            num_results: Number of results to return (max 100)
            country: Country code for localized results
            language: Language code for results
            time_filter: Time filter - "d" (day), "w" (week), "m" (month), "y" (year)
            config: Optional config for API key
        """
        self.query: str = query
        self.search_type: Literal["search", "news", "images"] = search_type
        self.num_results: int = min(num_results, 100)
        self.country: str = country
        self.language: str = language
        self.time_filter: str | None = time_filter

        # Get API key
        if config:
            api_key = config.serper.api_key
        else:
            from contextunity.router.core import get_core_config

            api_key = get_core_config().serper.api_key

        if not api_key:
            raise ConfigurationError("Serper API key not configured")
        self.api_key: str = api_key

    async def search_raw(self) -> list[dict[str, object]]:
        """Execute search and return raw results.

        Returns:
            List of result dictionaries with keys:
            - title: Result title
            - link: URL
            - snippet: Text snippet
            - date: Publication date (for news)
            - imageUrl: Image URL (for images)
        """
        endpoint = f"{self.BASE_URL}/{self.search_type}"

        payload: JsonDict = {
            "q": self.query,
            "gl": self.country,
            "hl": self.language,
            "num": self.num_results,
        }

        if self.time_filter:
            payload["tbs"] = f"qdr:{self.time_filter}"

        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                endpoint,
                json=payload,
                headers=headers,
            )
            _ = response.raise_for_status()
            data_raw: object = json_loads(response.text)

        if not is_json_dict(data_raw):
            return []

        # Extract results based on search type
        if self.search_type == "news":
            results_raw = data_raw.get("news", [])
        elif self.search_type == "images":
            results_raw = data_raw.get("images", [])
        else:
            results_raw = data_raw.get("organic", [])

        results = _serper_result_rows(results_raw)
        logger.debug("Serper returned %s results for '%s'", len(results), self.query)
        return results

    async def search(self) -> list[dict[str, str]]:
        """Execute search and return normalized results.

        Returns:
            List of normalized result dictionaries with keys:
            - title: Result title
            - url: URL
            - snippet: Text snippet
            - source: "serper"
        """
        raw_results = await self.search_raw()

        normalized: list[dict[str, str]] = []
        for item in raw_results:
            normalized.append(
                {
                    "title": as_str(item.get("title")),
                    "url": as_str(item.get("link")),
                    "snippet": as_str(item.get("snippet")),
                    "source": "serper",
                }
            )

        return normalized


__all__ = ["SerperSearchConnector"]
