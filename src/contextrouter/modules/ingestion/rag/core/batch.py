"""Unified batch processing utilities for LLM operations.

This module provides reusable helpers for batch LLM calls during ingestion:

**Key Functions**:

- `batch_validate(items, prompt_builder, ...)` - Validate items in batches,
  returns set of indices that passed validation. Used for filtering
  non-valuable content (QA questions/answers, video segments).

- `batch_transform(items, prompt_builder, result_parser, ...)` - Transform
  items in batches, returns dict mapping index to result. Used for
  generating summaries, extracting topics.

- `filter_by_indices(items, valid_indices)` - Filter sequence keeping only
  items at valid indices. Companion to batch_validate.

- `chunked(iterable, size)` - Yield successive chunks of given size.

**Design Patterns**:

1. Prompt builders receive `list[tuple[int, T]]` (indexed items) to enable
   TSV-style response parsing with index matching.

2. Error handling is configurable: `on_error="keep"` preserves items on
   LLM failure (safe default), `on_error="drop"` removes them.

3. Response parsing handles both real tabs and `<TAB>` literal markers
   (some LLMs output these instead of actual tab characters).

**Example Usage**:

```python
from contextrouter.modules.ingestion.rag.core.batch import batch_validate, filter_by_indices

def build_prompt(batch: list[tuple[int, str]]) -> str:
    items = "\\n".join(f"{idx}: {text}" for idx, text in batch)
    return f"Validate these items:\\n{items}\\n\\nReturn: idx<TAB>VALUABLE|NOT_VALUABLE"

valid_indices = batch_validate(
    records,
    prompt_builder=build_prompt,
    batch_size=50,
    on_error="keep",
)
filtered = filter_by_indices(records, valid_indices)
```
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from contextrouter.core import Config

from ..utils.llm import llm_generate

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class BatchResult(Generic[T]):
    """Result of batch processing with success/failure tracking."""

    successes: list[tuple[int, T]]  # (original_index, result)
    failures: list[int]  # original indices that failed

    @property
    def success_indices(self) -> set[int]:
        return {idx for idx, _ in self.successes}

    def get(self, idx: int, default: T | None = None) -> T | None:
        """Get result by original index."""
        for i, result in self.successes:
            if i == idx:
                return result
        return default


def batch_validate(
    items: Sequence[T],
    *,
    core_cfg: Config,
    prompt_builder: Callable[[list[tuple[int, T]]], str],
    batch_size: int = 50,
    model: str | None = None,
    parse_decision: Callable[[str], bool] = lambda s: "VALUABLE" in s.upper(),
    on_error: str = "keep",  # "keep" | "drop"
    label: str = "items",
) -> set[int]:
    """Batch validate items, returning indices of valid items.

    Args:
        items: Sequence of items to validate
        prompt_builder: Function that builds prompt from batch of (index, item) tuples
        batch_size: Items per LLM call
        model: LLM model to use
        parse_decision: Function to parse LLM output line into keep/drop decision
        on_error: What to do on LLM failure: "keep" all or "drop" all
        label: Label for logging

    Returns:
        Set of indices for items that passed validation
    """
    if not items:
        return set()

    valid_indices: set[int] = set()
    total = len(items)

    if not model:
        model = core_cfg.models.ingestion.preprocess.model.strip()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = [(i, items[i]) for i in range(batch_start, batch_end)]

        try:
            prompt = prompt_builder(batch)
            response = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=model,
                max_tokens=2048,
                temperature=0.0,
                parse_json=False,
            )

            if isinstance(response, str):
                _parse_validation_response(response, batch, valid_indices, parse_decision)
            elif on_error == "keep":
                valid_indices.update(i for i, _ in batch)

        except Exception as e:
            logger.warning(
                "Batch validation failed for %s at %d: %s (%s all)", label, batch_start, e, on_error
            )
            if on_error == "keep":
                valid_indices.update(i for i, _ in batch)

    dropped = total - len(valid_indices)
    if dropped:
        logger.info(
            "Validated %s: %d/%d passed (%d filtered)", label, len(valid_indices), total, dropped
        )

    return valid_indices


def _parse_validation_response(
    response: str,
    batch: list[tuple[int, Any]],
    valid_indices: set[int],
    parse_decision: Callable[[str], bool],
) -> None:
    """Parse TSV validation response and update valid_indices."""
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        parts = line.split("\t") if "\t" in line else line.split("<TAB>")
        if len(parts) < 2:
            continue

        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue

        decision = parts[1].strip()
        if parse_decision(decision):
            valid_indices.add(idx)


def batch_transform(
    items: Sequence[T],
    *,
    core_cfg: Config,
    prompt_builder: Callable[[list[tuple[int, T]]], str],
    result_parser: Callable[[str, int], R | None],
    batch_size: int = 50,
    model: str | None = None,
    label: str = "items",
) -> dict[int, R]:
    """Batch transform items, returning dict of index -> result.

    Args:
        items: Sequence of items to transform
        prompt_builder: Function that builds prompt from batch of (index, item) tuples
        result_parser: Function that parses a single line into result (or None to skip)
        batch_size: Items per LLM call
        model: LLM model to use
        label: Label for logging

    Returns:
        Dict mapping original index to transformed result
    """
    if not items:
        return {}

    results: dict[int, R] = {}
    total = len(items)

    if not model:
        model = core_cfg.models.ingestion.preprocess.model.strip()

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = [(i, items[i]) for i in range(batch_start, batch_end)]

        try:
            prompt = prompt_builder(batch)
            response = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=model,
                max_tokens=4096,
                temperature=0.1,
                parse_json=False,
            )

            if isinstance(response, str):
                for line in response.strip().splitlines():
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split("\t") if "\t" in line else line.split("<TAB>")
                    if not parts:
                        continue

                    try:
                        idx = int(parts[0].strip())
                    except ValueError:
                        continue

                    result = result_parser(line, idx)
                    if result is not None:
                        results[idx] = result

        except Exception as e:
            logger.warning("Batch transform failed for %s at %d: %s", label, batch_start, e)

    success_rate = len(results) / total if total else 0
    logger.debug("Transformed %s: %d/%d (%.0f%%)", label, len(results), total, success_rate * 100)

    return results


def chunked(iterable: Iterable[T], size: int) -> Iterable[list[T]]:
    """Yield successive chunks of given size from iterable.

    More memory-efficient than slicing for large iterables.
    """
    chunk: list[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def filter_by_indices(items: Sequence[T], valid_indices: set[int]) -> list[T]:
    """Filter sequence keeping only items at valid indices."""
    return [item for i, item in enumerate(items) if i in valid_indices]
