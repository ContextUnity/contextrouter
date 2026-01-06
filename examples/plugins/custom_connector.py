"""Example custom connector plugin for ContextRouter.

This shows how to add custom data connectors.
"""

from __future__ import annotations

from typing import Any

from contextrouter.core.registry import register_connector


@register_connector("custom_api")
class CustomAPIConnector:
    """Connector for a custom API source."""

    def __init__(self, **kwargs: Any):
        """Initialize the connector."""
        self.api_key = kwargs.get("api_key")
        self.endpoint = kwargs.get("endpoint", "https://api.example.com")

    def fetch(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Fetch data from the custom API."""
        # Your API fetching logic here
        return [
            {
                "title": f"Result for {query}",
                "content": f"Custom API data for: {query}",
                "source": "custom_api",
            }
        ]

    def test_connection(self) -> bool:
        """Test if the API connection works."""
        # Your connection test logic
        return True


print("Custom connector plugin loaded!")
