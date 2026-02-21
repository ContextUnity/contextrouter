"""
Lexicon node implementations.

Each node is a pure async function that takes LexiconState and returns state updates.
Follows the same patterns as Gardener nodes: LLM calls, JSON parsing, error handling.

Exception handling uses contextcore.exceptions hierarchy.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from contextcore.exceptions import ModelError, ProviderError

from .state import ContentRequest, GeneratedContent, LexiconState, ValidationResult

logger = logging.getLogger(__name__)


# --- Helpers ---


def _parse_json_response(content: str) -> Dict[str, Any]:
    """Extract JSON object from LLM response."""
    import re

    # Try to find JSON object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


# --- System Prompts ---

ANALYZE_PROMPT = """You are a product data analyst. Extract key features from raw product data.

Given the product information, identify:
1. Key selling features (3-5 items)
2. SEO-relevant keywords (5-8 items)
3. Product category in human-readable form

Respond in JSON:
```json
{
  "features": ["feature1", "feature2", ...],
  "seo_keywords": ["keyword1", "keyword2", ...],
  "category_label": "Human-readable category"
}
```

Language: {language}
"""

GENERATE_PROMPT = """You are a professional copywriter for an e-commerce platform.

Write a compelling product description based on the provided data.
Requirements:
- Title: short, SEO-optimized (max 80 chars)
- Description: informative, engaging (150-300 words)
- Include key features naturally in the text
- Tone: professional, modern, trustworthy
- Do NOT invent specifications not present in the data

Respond in JSON:
```json
{{
  "title": "Product title",
  "description": "Full product description...",
  "features": ["Feature 1", "Feature 2", ...]
}}
```

Language: {language}
"""


# --- Node Implementations ---


async def analyze_products_node(state: LexiconState) -> dict:
    """Fetch products and prepare content generation requests.

    Retrieves product data from Brain and builds ContentRequest objects.
    """
    start = time.time()
    product_ids = state.get("product_ids", [])

    if not product_ids:
        logger.info("Lexicon: no product_ids in state")
        return {"requests": [], "errors": ["No product_ids provided"]}

    try:
        from contextcore import BrainClient

        brain_url = state.get("brain_url", "localhost:50051")
        token = state.get("access_token")
        client = BrainClient(host=brain_url, mode="grpc", token=token)

        products_data = await client.get_products(
            tenant_id=state["tenant_id"],
            product_ids=product_ids,
            trace_id=state.get("trace_id", ""),
            parent_provenance=["router:lexicon:analyze"],
        )
    except Exception as e:
        logger.error("Lexicon: failed to fetch products from Brain: %s", e)
        raise ProviderError(
            f"Failed to fetch products from Brain: {e}",
            code="LEXICON_BRAIN_FETCH_ERROR",
        ) from e

    language = state.get("language", "uk")
    requests: List[ContentRequest] = []

    for p in products_data:
        requests.append(
            ContentRequest(
                product_id=p.get("id", 0),
                name=p.get("name", ""),
                category=p.get("category", ""),
                brand=p.get("brand_name", ""),
                description=p.get("description", ""),
                params=p.get("params", {}),
                language=language,
            )
        )

    logger.info("Lexicon: prepared %d content requests", len(requests))

    step_trace = {
        "step": "analyze",
        "products": len(requests),
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "requests": requests,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }


async def generate_content_node(state: LexiconState) -> dict:
    """Generate AI content for each product using LLM.

    Calls LLM with product data and generation prompt.
    Batches products to reduce API calls.
    """
    from ....llm import get_llm

    start = time.time()
    requests = state.get("requests", [])

    if not requests:
        logger.info("Lexicon: no requests to generate content for")
        return {"generated": []}

    llm = get_llm()
    language = state.get("language", "uk")
    generated: List[GeneratedContent] = []
    total_tokens = 0
    errors: List[str] = []

    # Process in batches of 5 to avoid overwhelming the LLM
    batch_size = 5
    for i in range(0, len(requests), batch_size):
        batch = requests[i : i + batch_size]

        for req in batch:
            product_info = json.dumps(
                {
                    "id": req.product_id,
                    "name": req.name,
                    "category": req.category,
                    "brand": req.brand,
                    "description": req.description,
                    "params": req.params,
                },
                ensure_ascii=False,
            )

            try:
                response = await llm.ainvoke(
                    [
                        {
                            "role": "system",
                            "content": GENERATE_PROMPT.format(language=language),
                        },
                        {"role": "user", "content": product_info},
                    ]
                )

                content = response.content if hasattr(response, "content") else str(response)
                parsed = _parse_json_response(content)
                tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
                total_tokens += tokens

                generated.append(
                    GeneratedContent(
                        product_id=req.product_id,
                        title=parsed.get("title", req.name),
                        description=parsed.get("description", ""),
                        features=parsed.get("features", []),
                        seo_keywords=parsed.get("seo_keywords", []),
                        language=language,
                        tokens_used=tokens,
                        model=getattr(llm, "model_name", "unknown"),
                    )
                )

            except Exception as e:
                logger.error(
                    "Lexicon: content generation failed for product %d: %s",
                    req.product_id,
                    e,
                )
                errors.append(f"generate:{req.product_id}:{e}")

    if not generated and errors:
        raise ModelError(
            f"All content generation failed: {len(errors)} errors",
            code="LEXICON_GENERATION_FAILED",
        )

    logger.info(
        "Lexicon: generated content for %d/%d products, tokens=%d",
        len(generated),
        len(requests),
        total_tokens,
    )

    step_trace = {
        "step": "generate",
        "products": len(generated),
        "errors": len(errors),
        "tokens": total_tokens,
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "generated": generated,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
        "errors": state.get("errors", []) + errors,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }


async def validate_content_node(state: LexiconState) -> dict:
    """Validate generated content against quality gates.

    Checks:
    - Title length (max 80 chars)
    - Description length (min 100 chars)
    - Features count (min 2)
    - No empty fields
    """
    start = time.time()
    generated = state.get("generated", [])

    if not generated:
        return {"validation": []}

    validation: List[ValidationResult] = []

    for item in generated:
        issues: List[str] = []

        if not item.title.strip():
            issues.append("Empty title")
        elif len(item.title) > 80:
            issues.append(f"Title too long: {len(item.title)} chars (max 80)")

        if not item.description.strip():
            issues.append("Empty description")
        elif len(item.description) < 100:
            issues.append(f"Description too short: {len(item.description)} chars (min 100)")

        if len(item.features) < 2:
            issues.append(f"Too few features: {len(item.features)} (min 2)")

        validation.append(
            ValidationResult(
                product_id=item.product_id,
                passed=len(issues) == 0,
                issues=issues,
            )
        )

    passed = sum(1 for v in validation if v.passed)
    failed = len(validation) - passed
    logger.info("Lexicon: validation — %d passed, %d failed", passed, failed)

    step_trace = {
        "step": "validate",
        "passed": passed,
        "failed": failed,
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "validation": validation,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }


async def write_results_node(state: LexiconState) -> dict:
    """Write validated content back to Brain via UpdateEnrichment RPC.

    Only writes content that passed validation.
    """
    start = time.time()

    generated = state.get("generated", [])
    validation = state.get("validation", [])

    # Build validation lookup
    valid_ids = {v.product_id for v in validation if v.passed}

    if not valid_ids:
        logger.warning("Lexicon: no content passed validation, nothing to write")
        return {
            "products_updated": 0,
            "errors": state.get("errors", []) + ["No content passed validation"],
        }

    try:
        from contextcore import BrainClient

        brain_url = state.get("brain_url", "localhost:50051")
        token = state.get("access_token")
        client = BrainClient(host=brain_url, mode="grpc", token=token)
    except Exception as e:
        raise ProviderError(
            f"Failed to connect to Brain for writing: {e}",
            code="LEXICON_BRAIN_CONNECT_ERROR",
        ) from e

    products_updated = 0
    errors: List[str] = []

    for item in generated:
        if item.product_id not in valid_ids:
            continue

        enrichment = {
            "lexicon": {
                "status": "done",
                "result": {
                    "title": item.title,
                    "description": item.description,
                    "features": item.features,
                    "seo_keywords": item.seo_keywords,
                    "language": item.language,
                    "model": item.model,
                    "tokens": item.tokens_used,
                },
            }
        }

        try:
            await client.update_enrichment(
                tenant_id=state["tenant_id"],
                product_id=item.product_id,
                enrichment=enrichment,
                trace_id=state.get("trace_id", ""),
                status="content_generated",
                parent_provenance=["router:lexicon:write_results"],
            )
            products_updated += 1

        except Exception as e:
            logger.error(
                "Lexicon: failed to write content for product %d: %s",
                item.product_id,
                e,
            )
            errors.append(f"write:{item.product_id}:{e}")

    logger.info("Lexicon: updated %d products", products_updated)

    step_trace = {
        "step": "write",
        "products": products_updated,
        "errors": len(errors),
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "products_updated": products_updated,
        "errors": state.get("errors", []) + errors,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }
