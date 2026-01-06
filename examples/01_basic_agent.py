"""Basic example of using the ContextRouter agent.

This script demonstrates how to initialize the agent and stream events from it.
"""

import asyncio

from contextrouter.cortex import stream_agent


async def main():
    # Example messages
    messages = [{"role": "user", "content": "What is the best way to implement RAG?"}]

    print("Starting ContextRouter agent...")

    # stream_agent is the main entry point for the "shared brain"
    async for event in stream_agent(
        messages=messages,
        session_id="example-session-123",
        platform="console",
        style_prompt="Be helpful and technical.",
    ):
        # Events follow the AG-UI protocol format
        event_type = event.get("event")
        data = event.get("data", {})

        if event_type == "text_delta":
            print(data.get("text", ""), end="", flush=True)
        elif event_type == "search_start":
            print(f"\n[Search] Started: {data.get('queries')}")
        elif event_type == "search_end":
            print(f"\n[Search] Completed with {len(data.get('citations', []))} citations")
        elif event_type == "error":
            print(f"\n[Error] {data.get('message')} (Code: {data.get('code')})")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
