"""
Lexicon state definitions.

State for the Lexicon subgraph: AI content generation for products.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class ContentRequest:
    """Single product content generation request."""

    product_id: int
    name: str
    category: str
    brand: str
    description: str
    params: Dict[str, Any]
    language: str = "uk"


@dataclass
class GeneratedContent:
    """LLM-generated content for a product."""

    product_id: int
    title: str = ""
    description: str = ""
    features: List[str] = field(default_factory=list)
    seo_keywords: List[str] = field(default_factory=list)
    language: str = "uk"
    tokens_used: int = 0
    model: str = ""


@dataclass
class ValidationResult:
    """Content validation outcome."""

    product_id: int
    passed: bool
    issues: List[str] = field(default_factory=list)


class LexiconState(TypedDict):
    """State for Lexicon subgraph.

    Flow:
        analyze → generate → validate → write_results
    """

    # Config (passed from CommerceState or runner)
    tenant_id: str
    brain_url: str
    language: str  # Target language (uk, en, ru)

    # Security
    access_token: Optional[Any]

    # Input
    product_ids: List[int]
    requests: List[ContentRequest]

    # Processing
    generated: List[GeneratedContent]
    validation: List[ValidationResult]

    # Trace
    trace_id: str
    step_traces: List[Dict[str, Any]]
    total_tokens: int

    # Output
    products_updated: int
    errors: List[str]
