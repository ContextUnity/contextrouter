"""Quick smoke test for RAG streaming via `stream_agent`.

This prints:
- token deltas if present in LangGraph events
- otherwise, prints the final assistant text on graph end

Usage:
  PYTHONPATH=src .venv/bin/python examples/91_test_rag_stream.py --query "What is X?"
"""

from __future__ import annotations

import argparse
import asyncio

from langchain_core.messages import HumanMessage

from contextrouter.cortex import stream_agent


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--session-id", default="local-test")
    args = ap.parse_args()

    full = ""
    async for event in stream_agent(
        messages=[HumanMessage(content=args.query)],
        session_id=args.session_id,
        platform="cli",
        enable_suggestions=False,
        enable_web_search=False,
    ):
        kind = event.get("event")

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            delta = getattr(chunk, "content", None)
            if isinstance(delta, str) and delta:
                full += delta
                print(delta, end="", flush=True)

        if kind in {"on_chain_end", "on_graph_end"} and not full:
            output = event.get("data", {}).get("output")
            msgs = output.get("messages") if isinstance(output, dict) else None
            last = msgs[-1] if isinstance(msgs, list) and msgs else None
            content = getattr(last, "content", None) if last is not None else None
            if isinstance(content, str) and content.strip():
                full = content
                print(content, end="", flush=True)

    print("\n")


if __name__ == "__main__":
    asyncio.run(main())
