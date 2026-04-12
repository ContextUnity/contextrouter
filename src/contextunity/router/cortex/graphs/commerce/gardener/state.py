"""
Gardener v2 state definitions.

Simplified state for a 3-node normalization pipeline:
  fetch_and_prepare → normalize → write_results
"""

from __future__ import annotations

from typing import Any, TypedDict


class GardenerState(TypedDict):
    """State for Gardener v2 normalization subgraph."""

    # --- Input (from payload) ---
    tenant_id: str
    brand: str  # Filter by brand name
    source: str  # "dealer" or "oscar" — MANDATORY, never mix!
    batch_size: int  # Default 50
    only_new: bool  # Only unprocessed products
    force: bool  # Re-normalize all products
    ids: list[int]  # Specific product IDs to normalize
    custom_hint: str  # Operator prompt hint for re-normalization
    execution_mode: str  # "sync" | "batch_submit" | "batch_status" | "batch_import"
    batch_job_id: str | None  # For querying/importing OpenAI Batch Jobs

    # --- Security ---
    access_token: Any | None  # ContextToken for authorization

    # --- Loaded data (set by fetch_and_prepare) ---
    taxonomy: dict[str, Any]  # categories + colors + sizes from YAML
    examples: list[dict[str, Any]]  # Few-shot examples (ONLY same source!)
    products: list[dict[str, Any]]  # Products to normalize

    # --- Output (set by normalize + write_results) ---
    results: list[dict[str, Any]]  # Normalization results
    taxonomy_candidates: list[dict[str, Any]]  # New values for taxonomy review
    stats: dict[str, Any]  # Counters and timings
    errors: list[str]  # Error messages
    batch_info: dict[str, Any]  # Returned from batch API (job_id, status, counts)

    # --- Trace ---
    trace_id: str
    step_traces: list[dict[str, Any]]
    total_tokens: int
