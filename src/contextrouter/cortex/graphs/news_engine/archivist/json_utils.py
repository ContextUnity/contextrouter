"""
JSON extraction utilities for LLM responses.

Handles various formats: raw JSON, markdown code blocks,
HTML entities, single quotes, trailing commas, etc.
"""

from __future__ import annotations

import json
import re


def extract_json_from_response(text: str) -> dict | None:
    """Extract JSON from LLM response, handling various formats.

    Handles:
    - Raw JSON objects
    - Markdown code blocks (```json ... ```)
    - JSON embedded in text
    - HTML entities (&quot; etc.)
    - Single quotes (Python dict format)
    - Unescaped newlines in strings
    - Trailing commas
    """
    if not text:
        return None

    # Try markdown code block first (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
        result = _try_parse_json(json_str)
        if result:
            return result

    # Replace HTML entities
    text = text.replace("&quot;", '"').replace("&#34;", '"')
    text = text.replace("&apos;", "'").replace("&#39;", "'")

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}") + 1

    if start >= 0 and end > start:
        json_str = text[start:end]
        return _try_parse_json(json_str)

    return None


def _try_parse_json(json_str: str) -> dict | None:
    """Try to parse JSON with multiple fallback strategies."""
    # Strategy 1: Direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Fix trailing commas
    fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Replace single quotes with double quotes (Python dict style)
    # Be careful not to replace apostrophes inside words
    fixed = re.sub(r"(?<![a-zA-Z])'([^']*)'(?![a-zA-Z])", r'"\1"', json_str)
    # Also handle keys with single quotes
    fixed = re.sub(r"'(\w+)':", r'"\1":', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Remove unescaped newlines in strings
    fixed = re.sub(r"(?<!\\)\n", " ", json_str)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Use ast.literal_eval for Python dict syntax
    try:
        import ast

        result = ast.literal_eval(json_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass

    return None


__all__ = ["extract_json_from_response"]
