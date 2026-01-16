"""LLM utilities for ingestion (no env access, no side effects)."""

from __future__ import annotations

import asyncio
import json
import logging

from contextrouter.core import Config
from contextrouter.modules.models.registry import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

logger = logging.getLogger(__name__)


def _resolve_json_model(core_cfg: Config, model: str) -> str:
    """Resolve model for JSON-critical ingestion steps via config override."""
    json_model = core_cfg.models.ingestion.json_model.model.strip()
    if json_model:
        return json_model
    return model


def llm_generate(
    *,
    core_cfg: Config,
    prompt: str,
    model: str,
    max_tokens: int = 16384,
    temperature: float = 0.1,
    max_retries: int = 5,
    parse_json: bool = True,
) -> dict[str, object] | list[object] | str:
    """Generate using a chat model (synchronous wrapper)."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            _llm_generate_impl(
                core_cfg=core_cfg,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
                parse_json=parse_json,
            )
        )

    # Running loop exists. This function is intentionally synchronous; do not nest event loops.
    raise RuntimeError(
        "llm_generate() is synchronous and cannot run inside an active asyncio loop. "
        "Call `await llm_generate_async(...)` instead."
    )


async def llm_generate_async(
    *,
    core_cfg: Config,
    prompt: str,
    model: str,
    max_tokens: int = 16384,
    temperature: float = 0.1,
    max_retries: int = 5,
    parse_json: bool = True,
) -> dict[str, object] | list[object] | str:
    """Async version of llm_generate(). Safe to call from within an event loop."""
    return await _llm_generate_impl(
        core_cfg=core_cfg,
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
        parse_json=parse_json,
    )


async def _llm_generate_impl(
    *,
    core_cfg: Config,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    parse_json: bool,
) -> dict[str, object] | list[object] | str:
    if parse_json:
        model = _resolve_json_model(core_cfg, model)

    for attempt in range(max_retries):
        try:
            model_instance = model_registry.get_llm_with_fallback(
                key=model,
                config=core_cfg,
            )

            request = ModelRequest(
                system="You are a helpful assistant specialized in structured data processing."
                if parse_json
                else None,
                parts=[TextPart(text=prompt)],
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            resp = await model_instance.generate(request)
            text = resp.text.strip()
            if not text:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Empty text, retrying (model=%s prompt_chars=%d attempt=%d/%d)",
                        model,
                        len(prompt),
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(3)
                    continue
                raise ValueError(
                    f"LLM returned empty text (model={model} prompt_chars={len(prompt)})"
                )

            if not parse_json:
                return text

            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            if not (text.startswith("{") or text.startswith("[")):
                import re

                if match := re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text):
                    text = match.group(1)

            try:
                result = json.loads(text)
                if not isinstance(result, (dict, list)):
                    raise ValueError(f"Expected dict or list, got {type(result)}")
                return result
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.warning("JSON parse failed, retrying...")
                    await asyncio.sleep(2)
                    continue
                raise ValueError(f"Failed to parse LLM JSON after {max_retries} attempts: {e}")

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 2**attempt * 10
                logger.warning("Rate limited, waiting %d seconds...", wait_time)
                await asyncio.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    raise ValueError("LLM generation failed after all retries")


__all__ = ["llm_generate", "llm_generate_async"]
