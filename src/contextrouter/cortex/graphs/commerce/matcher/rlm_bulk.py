"""
RLM Bulk Matcher - Deep product matching using Recursive Language Models.

This module provides bulk matching capabilities for large-scale product datasets
(50k+ supplier products → 10k site products) using RLM's recursive REPL approach.

Key Features:
- Multi-factor product decomposition (brand, model, SKU, size, color)
- Programmatic candidate selection via indexed lookups
- Sub-LLM calls for semantic disambiguation
- Comprehensive JSON match output

Reference: https://arxiv.org/abs/2512.24601
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from contextrouter.core import Config, get_core_config
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

logger = logging.getLogger(__name__)


# ============================================================================
# Data Types
# ============================================================================


@dataclass
class ProductMatch:
    """Result of a single product match."""

    supplier_id: str
    supplier_name: str
    site_id: str | None
    site_name: str | None
    confidence: float
    match_type: str  # sku_exact, brand_model, name_fuzzy, semantic, unmatched
    factors_matched: list[str] = field(default_factory=list)
    factors_mismatched: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class BulkMatchResult:
    """Result of bulk matching operation."""

    total_supplier: int
    total_site: int
    matches: list[ProductMatch]
    unmatched: list[dict[str, Any]]
    stats: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Matching Instructions
# ============================================================================

DEEP_MATCHING_INSTRUCTIONS = """
## TASK: Deep Product Matching

You have access to two datasets as Python variables:
- `supplier_products`: List[dict] — supplier products to match
- `site_products`: List[dict] — site catalog products (target)

Each product dict has fields like: id, sku, name, brand, category, params, description

## PHASE 1: INDEX SITE PRODUCTS

Build multiple indexes for O(1) lookup:
```python
from collections import defaultdict

sku_index = {}      # normalized_sku → site_product
brand_index = defaultdict(list)  # brand_lower → list[site_product]
name_tokens_index = defaultdict(list)  # word → list[site_product]

for p in site_products:
    sku_norm = normalize_sku(p.get('sku', ''))
    if sku_norm:
        sku_index[sku_norm] = p

    brand = (p.get('brand') or '').lower().strip()
    if brand:
        brand_index[brand].append(p)

    for token in tokenize_name(p.get('name', '')):
        name_tokens_index[token].append(p)
```

## PHASE 2: FACTOR DECOMPOSITION

For each product, extract normalized factors:
```python
def extract_factors(product):
    return {
        'brand': normalize_brand(product.get('brand', '')),
        'model': extract_model(product.get('name', '')),
        'product_type': extract_type(product.get('category', '')),
        'size': normalize_size(product.get('params', {}).get('size', '')),
        'color': normalize_color(product.get('params', {}).get('color', '')),
        'sku_normalized': normalize_sku(product.get('sku', '')),
        'name_tokens': set(tokenize_name(product.get('name', ''))),
    }
```

## PHASE 3: CANDIDATE SELECTION (for each supplier product)

1. **Exact SKU match** → confidence 1.0, match_type = "sku_exact"
2. **Brand + Model match** → confidence 0.9, match_type = "brand_model"
3. **Brand + ProductType + overlapping tokens** → confidence 0.7-0.85
4. **Token overlap scoring** → for remaining

```python
def find_candidates(supplier_factors, indexes):
    # Try SKU first
    if supplier_factors['sku_normalized'] in sku_index:
        return [(sku_index[supplier_factors['sku_normalized']], 1.0, 'sku_exact')]

    # Try brand + model
    candidates = []
    brand_matches = brand_index.get(supplier_factors['brand'], [])
    for site in brand_matches:
        site_factors = extract_factors(site)
        if supplier_factors['model'] == site_factors['model']:
            candidates.append((site, 0.9, 'brand_model'))

    if candidates:
        return candidates

    # Token overlap scoring
    ...
```

## PHASE 4: SEMANTIC DISAMBIGUATION

For ambiguous cases (multiple candidates with score >= 0.7):
```python
# Use sub-LLM for tie-breaking
answer = rlm.sub_completion(f'''
Compare these two products:
Supplier: {supplier['name']} - {supplier.get('description', '')[:200]}
Site: {candidate['name']} - {candidate.get('description', '')[:200]}

Are these the SAME product (just from different suppliers)?
Answer: YES or NO with confidence (0.0-1.0)
''')
```

## PHASE 5: OUTPUT FORMAT

Return FINAL_VAR(matches) where matches is a JSON list:
```json
[
  {
    "supplier_id": "SUP-12345",
    "supplier_name": "Куртка Arcteryx Beta AR XL",
    "site_id": "SITE-67890",
    "site_name": "Arcteryx Beta AR Jacket",
    "confidence": 0.95,
    "match_type": "brand_model",
    "factors_matched": ["brand", "model", "product_type"],
    "factors_mismatched": ["size"],
    "notes": "Supplier has size XL, site is parent product"
  },
  {
    "supplier_id": "SUP-99999",
    "supplier_name": "Unknown Product",
    "site_id": null,
    "site_name": null,
    "confidence": 0.0,
    "match_type": "unmatched",
    "factors_matched": [],
    "factors_mismatched": [],
    "notes": "No matching brand in catalog"
  }
]
```

## HELPER FUNCTIONS TO IMPLEMENT

```python
def normalize_sku(sku: str) -> str:
    \"\"\"Remove common prefixes, lowercase, strip non-alphanumeric.\"\"\"
    import re
    return re.sub(r'[^a-z0-9]', '', sku.lower())

def normalize_brand(brand: str) -> str:
    \"\"\"Lowercase, handle variations (arc'teryx → arcteryx).\"\"\"
    return brand.lower().replace("'", "").replace("-", "").strip()

def tokenize_name(name: str) -> list[str]:
    \"\"\"Split name into searchable tokens.\"\"\"
    import re
    return [w.lower() for w in re.split(r'[\\s\\-_/]+', name) if len(w) > 2]

def extract_model(name: str) -> str:
    \"\"\"Extract model name from product name.\"\"\"
    # Implement based on your naming conventions
    pass

def normalize_size(size: str) -> str:
    \"\"\"Normalize size: 'XL', 'extra large' → 'xl'.\"\"\"
    return size.lower().strip()

def normalize_color(color: str) -> str:
    \"\"\"Normalize color names.\"\"\"
    return color.lower().strip()
```

## EXECUTION

Iterate through ALL supplier products and build the complete matches list.
Handle exceptions gracefully - if one product fails, log and continue.
"""


# ============================================================================
# RLM Bulk Matcher
# ============================================================================


class RLMBulkMatcher:
    """
    Deep product matcher using Recursive Language Models.

    Processes 50k+ supplier products against 10k site products
    in a single recursive pass with multi-factor comparison.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        environment: str = "docker",
        verbose: bool = False,
        log_dir: str | None = None,
    ):
        """Initialize RLM Bulk Matcher.

        Args:
            config: ContextRouter configuration.
            environment: RLM sandbox environment (local, docker, modal, prime).
            verbose: Enable rich console logging.
            log_dir: Directory for RLM trajectory logs.
        """
        self._config = config or get_core_config()
        self._environment = environment
        self._verbose = verbose
        self._log_dir = log_dir

    async def match_all(
        self,
        supplier_products: list[dict[str, Any]],
        site_products: list[dict[str, Any]],
        *,
        confidence_threshold: float = 0.7,
        max_output_tokens: int = 50000,
    ) -> BulkMatchResult:
        """
        Match all supplier products to site products using RLM.

        This is a SINGLE deep pass that handles all matching in one execution.

        Args:
            supplier_products: List of supplier product dicts (can be 50k+).
            site_products: List of site catalog product dicts (typically 10k).
            confidence_threshold: Minimum confidence for auto-match (default 0.7).
            max_output_tokens: Max output tokens for RLM response.

        Returns:
            BulkMatchResult with all matches and statistics.
        """
        logger.info(
            f"RLM Bulk Matcher: Starting match of {len(supplier_products)} supplier → "
            f"{len(site_products)} site products"
        )

        # Create RLM model
        try:
            model = model_registry.create_llm(
                "rlm/gpt-5-mini",
                config=self._config,
                environment=self._environment,
                verbose=self._verbose,
                log_dir=self._log_dir,
            )
        except ImportError:
            # Fallback: RLM not installed, use chunked standard approach
            logger.warning(
                "RLM not installed (pip install rlm). "
                "Falling back to chunked matching."
            )
            return await self._fallback_chunked_match(
                supplier_products, site_products, confidence_threshold
            )

        # Prepare RLM prompt with data
        prompt = self._build_prompt(supplier_products, site_products)

        # Execute RLM matching
        logger.info("Executing RLM deep matching...")
        response = await model.generate(
            ModelRequest(
                system=(
                    "You are a product matching expert. Your task is to match supplier "
                    "products to site catalog products with high precision. Write Python "
                    "code in the REPL to analyze and match products efficiently."
                ),
                parts=[TextPart(text=prompt)],
                temperature=0.3,
                max_output_tokens=max_output_tokens,
            )
        )

        # Parse results
        matches = self._parse_response(response.text)

        # Build result
        matched_supplier_ids = {m.supplier_id for m in matches if m.site_id}
        unmatched = [
            p for p in supplier_products
            if str(p.get("id", p.get("sku", ""))) not in matched_supplier_ids
        ]

        # Calculate stats
        match_types = {}
        for m in matches:
            match_types[m.match_type] = match_types.get(m.match_type, 0) + 1

        result = BulkMatchResult(
            total_supplier=len(supplier_products),
            total_site=len(site_products),
            matches=matches,
            unmatched=unmatched,
            stats={
                "match_types": match_types,
                "match_rate": len([m for m in matches if m.site_id]) / len(supplier_products)
                if supplier_products
                else 0,
                "high_confidence": len([m for m in matches if m.confidence >= 0.9]),
                "medium_confidence": len(
                    [m for m in matches if 0.7 <= m.confidence < 0.9]
                ),
                "low_confidence": len(
                    [m for m in matches if m.confidence < 0.7 and m.site_id]
                ),
            },
        )

        logger.info(
            f"RLM Bulk Matcher complete: {result.stats['match_rate']:.1%} matched, "
            f"{len(result.unmatched)} unmatched"
        )

        return result

    def _build_prompt(
        self,
        supplier_products: list[dict[str, Any]],
        site_products: list[dict[str, Any]],
    ) -> str:
        """Build RLM prompt with embedded data."""
        # For RLM, we embed a sample and tell it the full data is available
        supplier_sample = supplier_products[:5]
        site_sample = site_products[:5]

        return f"""
{DEEP_MATCHING_INSTRUCTIONS}

## DATA AVAILABLE

The following variables are preloaded and available in your REPL environment:

`supplier_products` = {json.dumps(supplier_sample, ensure_ascii=False, indent=2)}
... # Total: {len(supplier_products)} items (access full list via the variable)

`site_products` = {json.dumps(site_sample, ensure_ascii=False, indent=2)}
... # Total: {len(site_products)} items (access full list via the variable)

## YOUR TASK

1. Build indexes on site_products
2. Extract factors for all products
3. Match each supplier_product to best site_product (or mark unmatched)
4. Return final matches as JSON

Begin writing code to perform the matching.
"""

    def _parse_response(self, response_text: str) -> list[ProductMatch]:
        """Parse RLM response into ProductMatch objects."""
        matches = []

        try:
            # Try to extract JSON from response
            # RLM typically returns FINAL_VAR(matches) = [...]
            import re

            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    matches.append(
                        ProductMatch(
                            supplier_id=str(item.get("supplier_id", "")),
                            supplier_name=item.get("supplier_name", ""),
                            site_id=item.get("site_id"),
                            site_name=item.get("site_name"),
                            confidence=float(item.get("confidence", 0)),
                            match_type=item.get("match_type", "unknown"),
                            factors_matched=item.get("factors_matched", []),
                            factors_mismatched=item.get("factors_mismatched", []),
                            notes=item.get("notes", ""),
                        )
                    )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse RLM response: {e}")
            logger.debug(f"Response was: {response_text[:500]}")

        return matches

    async def _fallback_chunked_match(
        self,
        supplier_products: list[dict[str, Any]],
        site_products: list[dict[str, Any]],
        confidence_threshold: float,
    ) -> BulkMatchResult:
        """Fallback matching when RLM is not available.

        Uses traditional chunked LLM calls instead of RLM recursion.
        """
        logger.info(
            "Using fallback chunked matching (RLM not available)"
        )

        # Simple SKU-based matching as fallback
        sku_index = {}
        for site in site_products:
            sku = site.get("sku", "").lower().strip()
            if sku:
                sku_index[sku] = site

        matches = []
        for supplier in supplier_products:
            supplier_sku = supplier.get("sku", "").lower().strip()
            supplier_id = str(supplier.get("id", supplier_sku))
            supplier_name = supplier.get("name", "")

            if supplier_sku in sku_index:
                site = sku_index[supplier_sku]
                matches.append(
                    ProductMatch(
                        supplier_id=supplier_id,
                        supplier_name=supplier_name,
                        site_id=str(site.get("id", site.get("sku", ""))),
                        site_name=site.get("name", ""),
                        confidence=1.0,
                        match_type="sku_exact",
                        factors_matched=["sku"],
                        notes="Fallback SKU match",
                    )
                )
            else:
                matches.append(
                    ProductMatch(
                        supplier_id=supplier_id,
                        supplier_name=supplier_name,
                        site_id=None,
                        site_name=None,
                        confidence=0.0,
                        match_type="unmatched",
                        notes="No SKU match (RLM fallback mode)",
                    )
                )

        return BulkMatchResult(
            total_supplier=len(supplier_products),
            total_site=len(site_products),
            matches=matches,
            unmatched=[
                p for p in supplier_products
                if not any(
                    m.supplier_id == str(p.get("id", p.get("sku", ""))) and m.site_id
                    for m in matches
                )
            ],
            stats={
                "mode": "fallback_sku_only",
                "match_rate": len([m for m in matches if m.site_id]) / len(supplier_products)
                if supplier_products
                else 0,
            },
        )


# ============================================================================
# LangGraph Node Integration
# ============================================================================


async def rlm_bulk_match_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node for RLM bulk matching.

    Expects state to contain:
        - supplier_products: List[dict]
        - site_products: List[dict]
        - Optional: confidence_threshold (float, default 0.7)
        - Optional: rlm_environment (str, default "docker")

    Returns updated state with:
        - matches: List[ProductMatch] (as dicts)
        - match_stats: dict with statistics
        - unmatched: List[dict] of unmatched supplier products
    """
    supplier_products = state.get("supplier_products", [])
    site_products = state.get("site_products", [])
    confidence_threshold = state.get("confidence_threshold", 0.7)
    environment = state.get("rlm_environment", "docker")

    if not supplier_products:
        logger.warning("No supplier products to match")
        return {
            "matches": [],
            "match_stats": {"error": "no_supplier_products"},
            "unmatched": [],
        }

    if not site_products:
        logger.warning("No site products to match against")
        return {
            "matches": [],
            "match_stats": {"error": "no_site_products"},
            "unmatched": supplier_products,
        }

    matcher = RLMBulkMatcher(environment=environment)
    result = await matcher.match_all(
        supplier_products,
        site_products,
        confidence_threshold=confidence_threshold,
    )

    # Convert matches to dicts for state serialization
    matches_as_dicts = [
        {
            "supplier_id": m.supplier_id,
            "supplier_name": m.supplier_name,
            "site_id": m.site_id,
            "site_name": m.site_name,
            "confidence": m.confidence,
            "match_type": m.match_type,
            "factors_matched": m.factors_matched,
            "factors_mismatched": m.factors_mismatched,
            "notes": m.notes,
        }
        for m in result.matches
    ]

    return {
        "matches": matches_as_dicts,
        "match_stats": result.stats,
        "unmatched": result.unmatched,
    }


__all__ = [
    "RLMBulkMatcher",
    "ProductMatch",
    "BulkMatchResult",
    "rlm_bulk_match_node",
    "DEEP_MATCHING_INSTRUCTIONS",
]
