"""Example of implementing and registering a custom data connector.

Connectors are responsible for fetching raw data from external sources.
"""

import asyncio
from typing import AsyncIterator

from contextcore import ContextUnit

from contextrouter.core.interfaces import BaseConnector
from contextrouter.core.registry import register_connector


@register_connector("my_custom_api")
class MyCustomConnector(BaseConnector):
    """A mock connector that simulates fetching data from a custom API."""

    def __init__(self, api_url: str = "https://api.example.com", **kwargs):
        self.api_url = api_url
        super().__init__()

    async def connect(self) -> AsyncIterator[ContextUnit]:
        # Simulate an API call
        print(f"Connecting to {self.api_url}...")
        await asyncio.sleep(0.5)

        # Yield some mocked data wrapped in a ContextUnit
        yield ContextUnit(
            payload={
                "title": "Custom Data Result",
                "text": "This is data fetched from a custom API connector.",
                "source": self.api_url,
            },
            provenance=["connector:my_custom_api"],
        )


async def main():
    from contextrouter.core.registry import create_connector

    # Instantiate via registry (how the framework does it)
    connector = create_connector("my_custom_api", params={"api_url": "https://api.martinell.ai"})

    print("Running custom connector...")
    async for unit in connector.connect():
        print(f"Received: {unit.payload}")
        print(f"Provenance: {unit.provenance}")


if __name__ == "__main__":
    asyncio.run(main())
