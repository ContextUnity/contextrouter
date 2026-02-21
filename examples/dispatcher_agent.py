"""Example usage of the always-active dispatcher agent.

This example demonstrates how to use the dispatcher agent via:
1. Python import (direct usage)
2. API calls (if running as a service)
"""

from __future__ import annotations

import asyncio

from contextrouter import get_dispatcher_service, invoke_dispatcher, stream_dispatcher


async def example_invoke() -> None:
    """Example: Invoke dispatcher agent (non-streaming)."""
    print("=== Example: Invoke Dispatcher Agent ===\n")

    messages = [{"role": "user", "content": "What tools are available in the system?"}]

    result = await invoke_dispatcher(
        messages=messages,
        session_id="example_session",
        platform="python",
    )

    print("Result:")
    print(result)
    print()


async def example_stream() -> None:
    """Example: Stream results from dispatcher agent."""
    print("=== Example: Stream Dispatcher Agent ===\n")

    messages = [
        {"role": "user", "content": "Search for information about Python async programming"}
    ]

    print("Streaming events:")
    async for event in stream_dispatcher(
        messages=messages,
        session_id="example_session",
        platform="python",
    ):
        print(f"Event: {event}")
    print()


async def example_service() -> None:
    """Example: Direct service access."""
    print("=== Example: Direct Service Access ===\n")

    service = get_dispatcher_service()

    # Access the graph directly
    graph = service.graph
    print(f"Graph type: {type(graph)}")

    # Invoke via service
    messages = [{"role": "user", "content": "Hello, dispatcher!"}]

    result = await service.invoke(
        messages=messages,
        session_id="example_service",
        platform="python",
    )

    print("Service result:")
    print(result)
    print()


async def main() -> None:
    """Run all examples."""
    await example_invoke()
    await example_stream()
    await example_service()


if __name__ == "__main__":
    asyncio.run(main())
