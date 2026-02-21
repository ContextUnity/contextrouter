"""
Gardener state definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class Product:
    """Product to enrich from DealerProduct."""

    id: int
    name: str
    category: str
    description: str
    params: Dict[str, Any]
    enrichment: Dict[str, Any]
    brand_name: Optional[str] = None

    def needs_task(self, task: str) -> bool:
        """Check if this task needs to be done."""
        task_data = self.enrichment.get(task, {})
        status = task_data.get("status")
        return status in (None, "pending", "error")


@dataclass
class EnrichmentResult:
    """Result of an enrichment task."""

    product_id: int
    task: str  # taxonomy, ner, params, tech, kg
    status: str  # done, error
    result: Dict[str, Any]
    tokens: int = 0
    error: Optional[str] = None


class GardenerState(TypedDict):
    """State for Gardener subgraph."""

    # Config (passed from CommerceState)
    batch_size: int
    brain_url: str  # Brain gRPC endpoint (e.g., "brain.contextunity.ts.net:50051")
    tenant_id: str
    prompts_dir: str

    # Security
    access_token: Optional[Any]  # ContextToken for authorization (from Router)

    # Products to process
    products: List[Product]

    # Results per task
    taxonomy_results: List[EnrichmentResult]
    ner_results: List[EnrichmentResult]
    params_results: List[EnrichmentResult]
    tech_results: List[EnrichmentResult]
    kg_results: List[EnrichmentResult]

    # Trace
    trace_id: str
    step_traces: List[Dict[str, Any]]
    total_tokens: int

    # Output
    products_updated: int
    errors: List[str]
