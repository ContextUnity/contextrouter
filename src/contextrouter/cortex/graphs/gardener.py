"""
Gardener Agent - LangGraph implementation.

The Ontologist: Enriches products with taxonomy, NER, parameters, technologies, KG.

Data model:
- Reads DealerProduct from Commerce DB
- Updates DealerProduct.enrichment JSON field
- Creates relations in Brain (knowledge_edges)

Flow:
1. fetch_pending - Get products where enrichment tasks are pending
2. classify_taxonomy - LLM taxonomy classification (batch)
3. extract_ner - LLM NER: product_type, brand, model (batch)
4. extract_params - LLM parameter extraction (batch by category)
5. extract_tech - LLM technology extraction (per item)
6. update_kg - Write relations to Brain knowledge_edges
7. write_results - Update DealerProduct.enrichment
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


# --- Data Types ---


@dataclass
class Product:
    """Product to enrich from DealerProduct."""

    id: int
    name: str
    category: str
    description: str
    params: Dict[str, Any]
    enrichment: Dict[str, Any]  # Current enrichment state

    # Extra fields for processing
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
    """State for Gardener graph."""

    # Config (passed from caller, NOT read from os.environ)
    batch_size: int
    db_url: str
    tenant_id: str
    prompts_dir: str

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


# --- Database Client ---


class DBClient:
    """Single client for both Commerce and Brain (same DB)."""

    def __init__(self, db_url: str):
        self.db_url = db_url

    async def get_pending_products(self, limit: int = 50) -> List[Product]:
        """Get products needing enrichment."""
        import psycopg
        from psycopg.rows import dict_row

        async with await psycopg.AsyncConnection.connect(self.db_url, row_factory=dict_row) as conn:
            async with conn.cursor() as cur:
                # Products where any enrichment task is pending
                await cur.execute(
                    """
                    SELECT
                        id, name, category, description, params, enrichment,
                        b.name as brand_name
                    FROM harvester_dealer_product dp
                    LEFT JOIN harvester_brand b ON dp.brand_id = b.id
                    WHERE
                        dp.status IN ('raw', 'enriching')
                        AND (
                            enrichment->>'taxonomy' IS NULL
                            OR enrichment->'taxonomy'->>'status' IN ('pending', 'error')
                            OR enrichment->>'ner' IS NULL
                            OR enrichment->'ner'->>'status' IN ('pending', 'error')
                        )
                    ORDER BY dp.created_at
                    LIMIT %s
                """,
                    (limit,),
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
        self, product_id: int, enrichment: Dict[str, Any], trace_id: str, status: str = "enriching"
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
        product_type: Optional[str],
        model_name: Optional[str],
        brand_id: Optional[int] = None,
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


# --- Node Implementations ---


async def fetch_pending_node(state: GardenerState) -> GardenerState:
    """Fetch products needing enrichment from queue.

    Products are enqueued by:
    - Worker scheduler (periodic)
    - Commerce AG-UI (user action)
    """
    from ..queue import get_enrichment_queue

    # Get product IDs from Redis queue
    queue = get_enrichment_queue()
    product_ids = await queue.dequeue(
        tenant_id=state["tenant_id"],
        batch_size=state["batch_size"],
        batch_id=state["trace_id"],
    )

    if not product_ids:
        logger.info("No products in enrichment queue")
        state["products"] = []
        return state

    # Load full product data from DB
    client = DBClient(state["db_url"])
    products = await client.get_products_by_ids(product_ids)

    state["products"] = products
    logger.info(f"Fetched {len(products)} products from queue for enrichment")

    return state


async def classify_taxonomy_node(state: GardenerState) -> GardenerState:
    """Classify taxonomy using LLM (batched)."""
    from ..llm import get_llm

    start = time.time()

    # Filter products needing taxonomy
    products = [p for p in state["products"] if p.needs_task("taxonomy")]

    if not products:
        logger.info("No products need taxonomy classification")
        return state

    llm = get_llm()

    # Build prompt
    items = [
        {"id": p.id, "name": p.name, "category": p.category}
        for p in products[:50]  # Limit batch
    ]

    # Load prompt from external file (no hardcoded business logic)
    system_prompt = _load_prompt(state["prompts_dir"], "taxonomy_classification.txt")

    try:
        response = await llm.ainvoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ]
        )

        # Parse response
        content = response.content if hasattr(response, "content") else str(response)
        # Extract JSON from response
        results = _parse_json_response(content)
        tokens = response.usage.total_tokens if hasattr(response, "usage") else 0

        # Create enrichment results
        for r in results:
            state["taxonomy_results"].append(
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

        state["total_tokens"] += tokens

    except Exception as e:
        logger.error(f"Taxonomy classification failed: {e}")
        state["errors"].append(f"taxonomy: {str(e)}")

    # Record trace
    state["step_traces"].append(
        {
            "step": "taxonomy",
            "products": len(products),
            "tokens": state.get("total_tokens", 0),
            "duration_ms": int((time.time() - start) * 1000),
        }
    )

    return state


async def extract_ner_node(state: GardenerState) -> GardenerState:
    """Extract NER (product_type, brand, model) using LLM (batched)."""
    from ..llm import get_llm

    start = time.time()

    # Filter products needing NER
    products = [p for p in state["products"] if p.needs_task("ner")]

    if not products:
        logger.info("No products need NER extraction")
        return state

    llm = get_llm()

    items = [{"id": p.id, "name": p.name, "brand_hint": p.brand_name} for p in products[:50]]

    # Load prompt from external file (no hardcoded business logic)
    system_prompt = _load_prompt(state["prompts_dir"], "ner_extraction.txt")

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

        for r in results:
            state["ner_results"].append(
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

        state["total_tokens"] += tokens

    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        state["errors"].append(f"ner: {str(e)}")

    state["step_traces"].append(
        {"step": "ner", "products": len(products), "duration_ms": int((time.time() - start) * 1000)}
    )

    return state


async def update_kg_node(state: GardenerState) -> GardenerState:
    """Create Knowledge Graph relations from NER results."""
    from pathlib import Path

    start = time.time()

    # Load ontology
    ontology_path = Path(__file__).parent.parent / "ontology" / "relations.json"
    if ontology_path.exists():
        with open(ontology_path) as f:
            ontology = json.load(f)
    else:
        ontology = {}

    client = DBClient(state["db_url"])
    tenant_id = state["tenant_id"]  # From config, NOT os.environ

    relations_created = 0

    for ner in state["ner_results"]:
        if ner.status != "done":
            continue

        product_id = str(ner.product_id)
        result = ner.result

        # Create MADE_BY relation
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

        # Create USES relations for technologies
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

        # Track relations in result
        state["kg_results"].append(
            EnrichmentResult(
                product_id=ner.product_id,
                task="kg",
                status="done",
                result={"relations_created": relations_created},
            )
        )

    logger.info(f"Created {relations_created} KG relations")

    state["step_traces"].append(
        {
            "step": "kg",
            "relations": relations_created,
            "duration_ms": int((time.time() - start) * 1000),
        }
    )

    return state


async def write_results_node(state: GardenerState) -> GardenerState:
    """Write all enrichment results back to DealerProduct."""
    start = time.time()

    client = DBClient(state["db_url"])
    now = datetime.now(timezone.utc).isoformat()

    # Merge all results by product_id
    results_by_product: Dict[int, Dict[str, Any]] = {}

    for result in state["taxonomy_results"] + state["ner_results"] + state["kg_results"]:
        if result.product_id not in results_by_product:
            results_by_product[result.product_id] = {}

        results_by_product[result.product_id][result.task] = {
            "status": result.status,
            "result": result.result,
            "tokens": result.tokens,
            "at": now,
            "error": result.error,
        }

    # Update each product
    for product in state["products"]:
        if product.id not in results_by_product:
            continue

        # Merge new results with existing enrichment
        enrichment = product.enrichment.copy()
        enrichment.update(results_by_product[product.id])

        # Determine status based on enrichment completion
        all_done = all(
            enrichment.get(task, {}).get("status") == "done" for task in ["taxonomy", "ner"]
        )
        status = "enriched" if all_done else "enriching"

        # If all done, set to pending_approval
        if all_done:
            status = "pending_approval"

        await client.update_product_enrichment(
            product_id=product.id, enrichment=enrichment, trace_id=state["trace_id"], status=status
        )

        # Also update direct fields from NER
        ner_result = enrichment.get("ner", {}).get("result", {})
        if ner_result:
            await client.update_product_ner_fields(
                product_id=product.id,
                product_type=ner_result.get("product_type"),
                model_name=ner_result.get("model"),
            )

        # Mark as done in queue
        from ..queue import get_enrichment_queue

        queue = get_enrichment_queue()
        await queue.mark_done(product.id, state["tenant_id"], state["trace_id"])

        state["products_updated"] += 1

    # Complete batch in queue
    from ..queue import get_enrichment_queue

    queue = get_enrichment_queue()
    await queue.complete_batch(state["trace_id"])

    state["step_traces"].append(
        {
            "step": "write",
            "products": state["products_updated"],
            "duration_ms": int((time.time() - start) * 1000),
        }
    )

    return state


# --- Helpers ---


def _parse_json_response(content: str) -> List[Dict]:
    """Extract JSON array from LLM response."""
    import re

    # Try to find JSON array in response
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try full content
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
    """Load prompt template from file.

    Prompts are external to keep business logic out of package.
    """
    from pathlib import Path

    path = Path(prompts_dir) / filename
    if path.exists():
        return path.read_text()

    # Fallback to package prompts
    package_path = Path(__file__).parent.parent / "prompts" / filename
    if package_path.exists():
        return package_path.read_text()

    raise FileNotFoundError(f"Prompt not found: {filename}")


# --- Graph Definition ---


def create_gardener_graph() -> StateGraph:
    """Create Gardener LangGraph."""
    graph = StateGraph(GardenerState)

    graph.add_node("fetch_pending", fetch_pending_node)
    graph.add_node("classify_taxonomy", classify_taxonomy_node)
    graph.add_node("extract_ner", extract_ner_node)
    graph.add_node("update_kg", update_kg_node)
    graph.add_node("write_results", write_results_node)

    graph.add_edge(START, "fetch_pending")
    graph.add_edge("fetch_pending", "classify_taxonomy")
    graph.add_edge("classify_taxonomy", "extract_ner")
    graph.add_edge("extract_ner", "update_kg")
    graph.add_edge("update_kg", "write_results")
    graph.add_edge("write_results", END)

    return graph.compile()


# --- Public API ---


async def invoke_gardener(
    batch_size: int = None,
    db_url: str = None,
    tenant_id: str = None,
    prompts_dir: str = None,
) -> Dict[str, Any]:
    """
    Run Gardener enrichment agent.

    Args:
        batch_size: Max products to process (from config if None)
        db_url: Database URL (from config if None)
        tenant_id: Tenant ID (from config if None)
        prompts_dir: Path to prompts directory (from config if None)

    Returns:
        {
            "processed": int,
            "updated": int,
            "tokens": int,
            "duration_ms": int,
            "step_traces": [...],
            "errors": [...]
        }
    """
    import uuid

    from ...core.config import get_core_config

    # Load config
    config = get_core_config()
    gardener_config = config.gardener

    # Use explicit args or fall back to config
    _batch_size = batch_size if batch_size is not None else gardener_config.batch_size
    _db_url = db_url or config.postgres.url
    _tenant_id = tenant_id or gardener_config.tenant_id
    # prompts_dir MUST be passed from caller (Worker) - commerce-specific
    _prompts_dir = prompts_dir

    if not _db_url:
        raise ValueError("Database URL not configured (postgres.url or db_url arg)")

    if not _tenant_id:
        raise ValueError("Tenant ID not configured (gardener.tenant_id)")

    if not _prompts_dir:
        raise ValueError("prompts_dir must be passed from caller (commerce-specific)")

    graph = create_gardener_graph()

    trace_id = (
        f"gardener:{uuid.uuid4().hex[:8]}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )

    initial_state: GardenerState = {
        "batch_size": _batch_size,
        "db_url": _db_url,
        "tenant_id": _tenant_id,
        "prompts_dir": _prompts_dir,
        "products": [],
        "taxonomy_results": [],
        "ner_results": [],
        "params_results": [],
        "tech_results": [],
        "kg_results": [],
        "trace_id": trace_id,
        "step_traces": [],
        "total_tokens": 0,
        "products_updated": 0,
        "errors": [],
    }

    start = time.time()
    final_state = await graph.ainvoke(initial_state)
    duration_ms = int((time.time() - start) * 1000)

    return {
        "trace_id": trace_id,
        "processed": len(final_state["products"]),
        "updated": final_state["products_updated"],
        "tokens": final_state["total_tokens"],
        "duration_ms": duration_ms,
        "step_traces": final_state["step_traces"],
        "errors": final_state["errors"],
    }
