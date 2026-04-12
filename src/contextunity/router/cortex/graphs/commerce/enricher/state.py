"""
Enricher Graph state definitions.

Handles the Product Creation and Enrichment flow via BiDi.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class ProductEnricherState(TypedDict):
    """State for Product Enrichment subgraph."""

    # --- Input (from payload) ---
    tenant_id: str
    dealer_products: List[Dict[str, Any]]

    # --- Auth & Configs injected by Shield/Init ---
    perplexity_api_key: str
    commerce_bidi_token: str

    # --- Internal Node Payloads ---
    normalized_data: List[Dict[str, Any]]  # The parsed out table data
    google_search_urls: Dict[int, List[str]]  # dealer_product_id -> list of urls
    descriptions: Dict[int, Dict[str, str]]  # dealer_product_id -> {uk: "", en: ""}
    seo_metadata: Dict[int, Dict[str, str]]

    # --- Technologies NER State ---
    extracted_technologies_names: List[str]
    missing_technologies: List[str]
    created_technologies_ids: List[int]

    # --- Output ready for Commerce Push ---
    mapped_attributes: Dict[int, List[Dict[str, Any]]]

    # --- Trace & Metrics ---
    trace_id: str
    step_traces: List[Dict[str, Any]]
    total_tokens: int
    errors: List[str]
