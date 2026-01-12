"""Quick smoke test for a single model key (generate + stream).

Usage:
  PYTHONPATH=src .venv/bin/python examples/90_test_model_generate.py --model openai/gpt-5.1

Notes:
  - For OpenAI/OpenRouter/local providers: install extras `contextrouter[models]`
  - For HF transformers: install extras `contextrouter[hf-transformers]`
"""

from __future__ import annotations

import argparse
import asyncio

from contextrouter.core.config import Config
from contextrouter.modules.models.registry import model_registry
from contextrouter.modules.models.types import (
    FinalTextEvent,
    ModelRequest,
    TextDeltaEvent,
    TextPart,
)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model key, e.g. vertex/gemini-2.5-flash")
    ap.add_argument("--prompt", default="Say hello in one short sentence.")
    args = ap.parse_args()

    cfg = Config.load()
    model = model_registry.create_llm(args.model, config=cfg)
    req = ModelRequest(parts=[TextPart(text=args.prompt)])

    resp = await model.generate(req)
    print("\n=== generate() ===")
    print(resp.text)
    print("provider:", resp.raw_provider.model_key)

    print("\n=== stream() ===")
    async for ev in model.stream(req):
        if isinstance(ev, TextDeltaEvent):
            print(ev.delta, end="", flush=True)
        elif isinstance(ev, FinalTextEvent):
            print("\n\n[final]")
            print(ev.text)
            break


if __name__ == "__main__":
    asyncio.run(main())
