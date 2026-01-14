"""Example: Using KeyphraseTransformer in ContextRouter (not wired into graphs).

Run:
  uv run python examples/keyphrase_usage.py
"""

from __future__ import annotations

import asyncio

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.modules.transformers.keyphrases import KeyphraseTransformer


async def main() -> None:
    transformer = KeyphraseTransformer()
    transformer.configure({"mode": "llm", "max_phrases": 12, "min_score": 0.15})

    text = """
    ContextRouter is a modular, LangGraph-powered shared brain designed for high-performance
    agentic workflows. It uses the Bisquit protocol to wrap data in a BisquitEnvelope for provenance
    and security. The system separates orchestration (cortex) from modules/providers/connectors.
    """

    envelope = BisquitEnvelope(content={"content": text}, metadata={"source": "example"})
    enriched = await transformer.transform(envelope)

    keyphrases = enriched.metadata.get("keyphrases", [])
    print("=== Keyphrases ===")
    for kp in keyphrases:
        print(f"- {kp['text']} (score={kp['score']:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
