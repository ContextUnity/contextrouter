"""LLM utilities for ingestion (no env access, no side effects).

This module is library code. It must not:
- call `load_env()` at import time
- read `os.environ` directly
- instantiate provider SDK clients from ambient environment

Instead, the caller must pass a validated `contextrouter.core.config.Config` and an explicit
model registry key (`provider/name`, e.g. `vertex/gemini-2.5-pro`).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from contextrouter.core.config import Config
from contextrouter.modules.models.registry import model_registry

LOGGER = logging.getLogger(__name__)

MODEL_PRO = "vertex/gemini-2.5-pro"
MODEL_FLASH = "vertex/gemini-2.5-flash"
MODEL_LIGHT = "vertex/gemini-2.5-flash-lite"


def llm_generate(
    *,
    core_cfg: Config,
    prompt: str,
    model: str = MODEL_PRO,
    max_tokens: int = 16384,
    temperature: float = 0.1,
    max_retries: int = 5,
    parse_json: bool = True,
) -> dict[str, Any] | list[Any] | str:
    """Generate using a chat model created from the model registry.

    `model` is a registry key: `provider/name`. There is no implicit fallback.
    """

    for attempt in range(max_retries):
        try:
            # Get model with fallback support
            model_instance = model_registry.get_llm_with_fallback(
                key=model,
                config=core_cfg,
            )

            from contextrouter.modules.models.types import ModelRequest, TextPart

            request = ModelRequest(
                parts=[TextPart(text=prompt)],
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            resp = await model_instance.generate(request)
            text = resp.text
            text = text.strip()
            if not text:
                LOGGER.warning("Empty text in response, attempt %d/%d", attempt + 1, max_retries)
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                raise ValueError("LLM returned empty text")

            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 2:
                    text = parts[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

            if parse_json and not text.startswith("{") and not text.startswith("["):
                import re

                json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
                if json_match:
                    text = json_match.group(1)

            if not parse_json:
                return text

            try:
                result = json.loads(text)
                if not isinstance(result, (dict, list)):
                    raise ValueError(f"Expected dict or list, got {type(result)}")
                return result
            except json.JSONDecodeError as e:
                LOGGER.warning("JSON parse failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise ValueError(
                    f"Failed to parse LLM JSON response after {max_retries} attempts: {e}"
                )

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                wait_time = 2**attempt * 10
                LOGGER.warning("Rate limited, waiting %d seconds...", wait_time)
                time.sleep(wait_time)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    raise ValueError("LLM generation failed after all retries")
