"""Ingestion -> RAG roundtrip smoke test (CLI-driven).

This uses the existing ingestion CLI to build assets and then runs a RAG query.

Prereqs:
  - install ingestion extras: `pip install contextrouter[ingestion]`
  - prepare an ingestion config (see contextrouter-plan / docs)

Usage:
  PYTHONPATH=src .venv/bin/python examples/92_test_ingestion_roundtrip.py \
    --ingest-config path/to/ingest.toml \
    --query "What is ...?"
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage

from contextrouter.cortex import stream_agent


def run_ingestion(config_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "contextrouter.cli.app",
        "ingest",
        "run",
        "--config-path",
        str(config_path),
    ]
    subprocess.run(cmd, check=True)


async def run_rag(query: str) -> str:
    full = ""
    async for event in stream_agent(
        messages=[HumanMessage(content=query)],
        session_id="ingestion-roundtrip",
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
        if kind in {"on_chain_end", "on_graph_end"} and not full:
            output = event.get("data", {}).get("output")
            msgs = output.get("messages") if isinstance(output, dict) else None
            last = msgs[-1] if isinstance(msgs, list) and msgs else None
            content = getattr(last, "content", None) if last is not None else None
            if isinstance(content, str) and content.strip():
                full = content
    return full


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest-config", type=Path, required=True)
    ap.add_argument("--query", required=True)
    args = ap.parse_args()

    run_ingestion(args.ingest_config)
    answer = await run_rag(args.query)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
