"""
JSON parsing utilities for harvest responses.

Handles extracting JSON arrays from LLM responses that may contain
extra text, markdown code blocks, or malformed content.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


def extract_json_array(text: str) -> list[dict]:
    """Extract JSON array from LLM response.

    Handles:
    - Raw JSON arrays
    - JSON embedded in explanatory text
    - Markdown code blocks (```json ... ```)
    - Empty arrays []
    - Nested brackets

    Args:
        text: Raw response text from LLM

    Returns:
        Parsed list of dicts, or empty list if parsing failed
    """
    if not text or len(text.strip()) < 2:
        logger.warning(f"Empty or too short response: '{text[:100] if text else 'None'}'")
        return []

    logger.debug(f"Parsing JSON from response length: {len(text)}")

    # Step 1: Strip markdown code blocks first
    # This handles ```json ... ``` and ``` ... ```
    code_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if code_match:
        json_str = code_match.group(1).strip()
        try:
            result = json.loads(json_str)
            if isinstance(result, list):
                logger.debug(f"Parsed from code block: {len(result)} items")
                return result
        except json.JSONDecodeError:
            pass  # Continue to other methods

    try:
        # Step 2: Look for JSON array with objects - pattern: [ { ... } ]
        match = re.search(r"\[\s*\{", text)
        if match:
            start = match.start()
            # Find matching closing ]
            depth = 0
            end = start
            for i, char in enumerate(text[start:]):
                if char == "[":
                    depth += 1
                elif char == "]":
                    depth -= 1
                    if depth == 0:
                        end = start + i + 1
                        break

            json_str = text[start:end]
            logger.debug(f"Extracted JSON ({len(json_str)} chars): {json_str[:200]}...")
            return json.loads(json_str)

        # Step 3: Look for any JSON array (including empty [])
        array_match = re.search(r"\[[\s\S]*?\]", text)
        if array_match:
            json_str = array_match.group(0)
            try:
                result = json.loads(json_str)
                if isinstance(result, list):
                    logger.debug(f"Parsed array: {len(result)} items")
                    return result
            except json.JSONDecodeError:
                pass

        logger.warning("No JSON array found in response")
        return []

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}")
        return []


__all__ = ["extract_json_array"]
