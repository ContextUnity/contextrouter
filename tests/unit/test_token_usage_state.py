"""Tests for GraphState ``_token_usage`` reducer."""

from __future__ import annotations

from contextunity.router.cortex.types import _merge_token_usage


def test_merge_token_usage_prefers_latest_right() -> None:
    left = {"input_tokens": 10, "output_tokens": 5, "total_cost": 0.001}
    right = {"input_tokens": 110, "output_tokens": 55, "total_cost": 0.011}
    merged = _merge_token_usage(left, right)
    assert merged == right


def test_merge_token_usage_keeps_left_when_right_empty() -> None:
    left = {"input_tokens": 10, "output_tokens": 5}
    assert _merge_token_usage(left, {}) == left
