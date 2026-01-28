"""
Gardener node implementations.

Each node is a pure async function that takes state and returns state updates.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List

from .state import EnrichmentResult, GardenerState, Product

logger = logging.getLogger(__name__)


# --- Database Client ---


class DBClient:
    """Single client for both Commerce and Brain (same DB)."""

    def __init__(self, db_url: str):
        self.db_url = db_url

    async def get_products_by_ids(self, product_ids: List[int]) -> List[Product]:
        """Get products by IDs (for queue-based processing)."""
        if not product_ids:
            return []

        import psycopg
        from psycopg.rows import dict_row

        async with await psycopg.AsyncConnection.connect(self.db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        dp.id, dp.name, dp.category, dp.description, dp.params, dp.enrichment,
                        b.name as brand_name
                    FROM harvester_dealer_product dp
                    LEFT JOIN harvester_brand b ON dp.brand_id = b.id
                    WHERE dp.id = ANY(%s)
                """,
                    (product_ids,),
                )

                rows = await cur.fetchall()
                return [
                    Product(
                        id=row["id"],
                        name=row["name"] or "",
                        category=row["category"] or "",
                        description=row["description"] or "",
                        params=row["params"] or {},
                        enrichment=row["enrichment"] or {},
                        brand_name=row["brand_name"],
                    )
                    for row in rows
                ]

    async def update_product_enrichment(
        self,
        product_id: int,
        enrichment: Dict[str, Any],
        trace_id: str,
        status: str = "enriching",
    ) -> None:
        """Update product enrichment field."""
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE harvester_dealer_product
                    SET
                        enrichment = %s,
                        enrichment_trace_id = %s,
                        status = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """,
                    (json.dumps(enrichment), trace_id, status, product_id),
                )
                await conn.commit()

    async def update_product_ner_fields(
        self,
        product_id: int,
        product_type: str | None,
        model_name: str | None,
        brand_id: int | None = None,
    ) -> None:
        """Update NER-extracted fields on product."""
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE harvester_dealer_product
                    SET
                        product_type = COALESCE(%s, product_type),
                        model_name = COALESCE(%s, model_name),
                        updated_at = NOW()
                    WHERE id = %s
                """,
                    (product_type, model_name, product_id),
                )
                await conn.commit()

    async def create_kg_relation(
        self,
        tenant_id: str,
        source_type: str,
        source_id: str,
        relation: str,
        target_type: str,
        target_id: str,
    ) -> None:
        """Create relation in knowledge_edges."""
        import psycopg

        async with await psycopg.AsyncConnection.connect(self.db_url) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO knowledge_edges
                        (tenant_id, source_type, source_id, relation, target_type, target_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT DO NOTHING
                """,
                    (tenant_id, source_type, source_id, relation, target_type, target_id),
                )
                await conn.commit()


# --- Helpers ---


def _parse_json_response(content: str) -> List[Dict]:
    """Extract JSON array from LLM response."""
    import re

    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        return [result]
    except json.JSONDecodeError:
        return []


def _slugify(text: str) -> str:
    """Convert text to slug."""
    import re

    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text


def _load_prompt(prompts_dir: str, filename: str) -> str:
    """Load prompt template from file."""
    from pathlib import Path

    path = Path(prompts_dir) / filename
    if path.exists():
        return path.read_text()

    package_path = Path(__file__).parent.parent.parent / "prompts" / filename
    if package_path.exists():
        return package_path.read_text()

    raise FileNotFoundError(f"Prompt not found: {filename}")


# --- Node Implementations ---


async def fetch_pending_node(state: GardenerState) -> dict:
    """Fetch products needing enrichment from queue."""
    from ....queue import get_enrichment_queue

    queue = get_enrichment_queue()
    product_ids = await queue.dequeue(
        tenant_id=state["tenant_id"],
        batch_size=state["batch_size"],
        batch_id=state["trace_id"],
    )

    if not product_ids:
        logger.info("No products in enrichment queue")
        return {"products": []}

    client = DBClient(state["db_url"])
    products = await client.get_products_by_ids(product_ids)

    logger.info(f"Fetched {len(products)} products from queue for enrichment")
    return {"products": products}


async def classify_taxonomy_node(state: GardenerState) -> dict:
    """Classify taxonomy using LLM (batched)."""
    from ....llm import get_llm

    start = time.time()
    products = [p for p in state["products"] if p.needs_task("taxonomy")]

    if not products:
        logger.info("No products need taxonomy classification")
        return {}

    llm = get_llm()
    items = [{"id": p.id, "name": p.name, "category": p.category} for p in products[:50]]
    system_prompt = _load_prompt(state["prompts_dir"], "taxonomy_classification.txt")

    taxonomy_results = []
    total_tokens = 0

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ]
        )

        content = response.content if hasattr(response, "content") else str(response)
        results = _parse_json_response(content)
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        total_tokens = tokens

        for r in results:
            taxonomy_results.append(
                EnrichmentResult(
                    product_id=r["id"],
                    task="taxonomy",
                    status="done",
                    result={
                        "category": r.get("category"),
                        "color": r.get("color"),
                        "size": r.get("size"),
                        "gender": r.get("gender"),
                        "confidence": r.get("confidence", 0.8),
                    },
                    tokens=tokens // len(results) if results else 0,
                )
            )

    except Exception as e:
        logger.error(f"Taxonomy classification failed: {e}")
        return {"errors": state.get("errors", []) + [f"taxonomy: {str(e)}"]}

    step_trace = {
        "step": "taxonomy",
        "products": len(products),
        "tokens": total_tokens,
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "taxonomy_results": state.get("taxonomy_results", []) + taxonomy_results,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }


async def extract_ner_node(state: GardenerState) -> dict:
    """Extract NER (product_type, brand, model) using LLM (batched)."""
    from ....llm import get_llm

    start = time.time()
    products = [p for p in state["products"] if p.needs_task("ner")]

    if not products:
        logger.info("No products need NER extraction")
        return {}

    llm = get_llm()
    items = [{"id": p.id, "name": p.name, "brand": p.brand_name or ""} for p in products[:50]]
    system_prompt = _load_prompt(state["prompts_dir"], "ner_extraction.txt")

    ner_results = []
    total_tokens = 0

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ]
        )

        content = response.content if hasattr(response, "content") else str(response)
        results = _parse_json_response(content)
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0
        total_tokens = tokens

        for r in results:
            ner_results.append(
                EnrichmentResult(
                    product_id=r["id"],
                    task="ner",
                    status="done",
                    result={
                        "product_type": r.get("product_type"),
                        "brand": r.get("brand"),
                        "model": r.get("model"),
                        "technologies": r.get("technologies", []),
                    },
                    tokens=tokens // len(results) if results else 0,
                )
            )

    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        return {"errors": state.get("errors", []) + [f"ner: {str(e)}"]}

    step_trace = {
        "step": "ner",
        "products": len(products),
        "tokens": total_tokens,
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "ner_results": state.get("ner_results", []) + ner_results,
        "total_tokens": state.get("total_tokens", 0) + total_tokens,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }


async def update_kg_node(state: GardenerState) -> dict:
    """Create Knowledge Graph relations from NER results."""
    from pathlib import Path

    start = time.time()

    ontology_path = Path(__file__).parent.parent.parent / "ontology" / "relations.json"
    if ontology_path.exists():
        with open(ontology_path) as f:
            ontology = json.load(f)
    else:
        ontology = {}

    client = DBClient(state["db_url"])
    tenant_id = state["tenant_id"]

    relations_created = 0
    kg_results = []

    for ner in state.get("ner_results", []):
        if ner.status != "done":
            continue

        product_id = str(ner.product_id)
        result = ner.result

        if result.get("brand") and "MADE_BY" in ontology:
            brand_slug = _slugify(result["brand"])
            await client.create_kg_relation(
                tenant_id=tenant_id,
                source_type="product",
                source_id=product_id,
                relation="MADE_BY",
                target_type="brand",
                target_id=brand_slug,
            )
            relations_created += 1

        for tech in result.get("technologies", []):
            if "USES" in ontology:
                tech_slug = _slugify(tech)
                await client.create_kg_relation(
                    tenant_id=tenant_id,
                    source_type="product",
                    source_id=product_id,
                    relation="USES",
                    target_type="technology",
                    target_id=tech_slug,
                )
                relations_created += 1

        kg_results.append(
            EnrichmentResult(
                product_id=ner.product_id,
                task="kg",
                status="done",
                result={"relations_created": relations_created},
            )
        )

    logger.info(f"Created {relations_created} KG relations")

    step_trace = {
        "step": "kg",
        "relations": relations_created,
        "duration_ms": int((time.time() - start) * 1000),
    }

    return {
        "kg_results": state.get("kg_results", []) + kg_results,
        "step_traces": state.get("step_traces", []) + [step_trace],
    }


async def write_results_node(state: GardenerState) -> dict:
    """Write all enrichment results back to DealerProduct."""
    start = time.time()

    client = DBClient(state["db_url"])
    products_updated = 0
    errors = []

    # Build enrichment map per product
    enrichment_map: Dict[int, Dict[str, Any]] = {}

    for result in (
        state.get("taxonomy_results", [])
        + state.get("ner_results", [])
        + state.get("kg_results", [])
    ):
        if result.product_id not in enrichment_map:
            product = next((p for p in state["products"] if p.id == result.product_id), None)
            enrichment_map[result.product_id] = product.enrichment.copy() if product else {}

        enrichment_map[result.product_id][result.task] = {
            "status": result.status,
            "result": result.result,
            "tokens": result.tokens,
        }

    # Write to DB
    for product_id, enrichment in enrichment_map.items():
        try:
            await client.update_product_enrichment(
                product_id=product_id,
                enrichment=enrichment,
                trace_id=state["trace_id"],
                status="enriched",
            )

            ner_result = next(
                (r for r in state.get("ner_results", []) if r.product_id == product_id),
                None,
            )
            if ner_result and ner_result.status == "done":
                await client.update_product_ner_fields(
                    product_id=product_id,
                    product_type=ner_result.result.get("product_type"),
                    model_name=ner_result.result.get("model"),
                )

            products_updated += 1

        except Exception as e:
            logger.error(f"Failed to write results for product {product_id}: {e}")
            errors.append(f"write:{product_id}:{str(e)}")

    logger.info(f"Updated {products_updated} products")

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
