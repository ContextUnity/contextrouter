"""
Serper Search Connector.

High-performance connector for Serper.dev Google Search API.
Supports web, news, images search with filtering options.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from contextrouter.core.config import Config

logger = logging.getLogger(__name__)


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

    BASE_URL = "https://google.serper.dev"

    def __init__(
        self,
        query: str,
        *,
        search_type: Literal["search", "news", "images"] = "search",
        num_results: int = 10,
        country: str = "us",
        language: str = "en",
        time_filter: str | None = None,
        config: Config | None = None,
    ):
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
        self.query = query
        self.search_type = search_type
        self.num_results = min(num_results, 100)
        self.country = country
        self.language = language
        self.time_filter = time_filter

        # Get API key
        if config:
            self.api_key = config.serper.api_key
        else:
            from contextrouter.core import get_core_config

            self.api_key = get_core_config().serper.api_key

        if not self.api_key:
            raise ValueError("Serper API key not configured")

    async def search_raw(self) -> list[dict[str, Any]]:
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

        payload: dict[str, Any] = {
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
            response.raise_for_status()
            data = response.json()

        # Extract results based on search type
        if self.search_type == "news":
            results = data.get("news", [])
        elif self.search_type == "images":
            results = data.get("images", [])
        else:
            results = data.get("organic", [])

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

        normalized = []
        for item in raw_results:
            normalized.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "serper",
                }
            )

        return normalized


__all__ = ["SerperSearchConnector"]
