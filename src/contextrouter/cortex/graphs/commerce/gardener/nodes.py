"""
Gardener v2 node implementations.

Three-node pipeline:
  fetch_and_prepare → normalize → write_results

Each node is created via make_* factory that bakes config into closures
(same pattern as sql_analytics nodes).
"""

from __future__ import annotations

import time

from contextcore import get_context_unit_logger

from .bidi import GardenerBiDi
from .normalizer import merge_results, run_deterministic_pass, run_llm_pass
from .state import GardenerState

logger = get_context_unit_logger(__name__)


# ── Node 1: fetch_and_prepare ──


def make_fetch_and_prepare():
    """Create fetch_and_prepare node."""

    async def fetch_and_prepare(state: GardenerState) -> dict:
        """Load taxonomy, find few-shot examples, fetch products to normalize."""
        import uuid

        start = time.time()
        brand = state.get("brand", "")
        source = state.get("source", "dealer")
        only_new = state.get("only_new", True)
        batch_size = state.get("batch_size", 50)
        tenant_id = state.get("tenant_id", "traverse")
        trace_id = state.get("trace_id", uuid.uuid4().hex[:12])

        bidi = GardenerBiDi(trace_id, tenant_id=tenant_id)
        execution_mode = state.get("execution_mode", "sync")

        if execution_mode in ("batch_status", "batch_import"):
            return {
                "trace_id": trace_id,
                "products": [],
                "taxonomy": {},
                "examples": [],
            }

        # 1. Load taxonomy
        taxonomy = await bidi.export_taxonomy()
        logger.info("Loaded taxonomy: %d keys", len(taxonomy))

        # 2. Find already-normalized examples - DISABLED to prevent passing poisoned examples from prior runs.
        examples = []
        logger.info("Examples disabled for robust LLM behavior")

        # 3. Fetch unprocessed products
        ids = state.get("ids", [])
        products = await bidi.export_products_for_normalization(
            brand=brand,
            source=source,
            only_new=only_new,
            batch_size=batch_size,
            ids=ids,
        )
        logger.info(
            "Fetched %d %s products for normalization (brand=%s)",
            len(products),
            source,
            brand,
        )

        step_trace = {
            "step": "fetch_and_prepare",
            "brand": brand,
            "source": source,
            "examples": len(examples),
            "products": len(products),
            "duration_ms": int((time.time() - start) * 1000),
        }

        return {
            "taxonomy": taxonomy,
            "examples": examples,
            "products": products,
            "trace_id": trace_id,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return fetch_and_prepare


# ── Node 2: normalize ──


def make_normalize(
    *,
    model_key: str,
    reasoning_effort: str = "none",
):
    """Create normalize node with model config baked in.

    Args:
        model_key: LLM model key (e.g. 'openai/gpt-5-nano')
        reasoning_effort: LLM reasoning effort level
    """

    async def normalize(state: GardenerState) -> dict:
        """Two-pass normalization: deterministic + LLM."""
        start = time.time()
        products = state.get("products", [])
        taxonomy = state.get("taxonomy", {})
        examples = state.get("examples", [])
        custom_hint = state.get("custom_hint", "")
        total_tokens = 0

        if not products:
            logger.info("No products to normalize")
            return {"results": [], "taxonomy_candidates": []}

        # ── Pass 1: Deterministic color/size resolution ──
        deterministic_results = run_deterministic_pass(products, taxonomy)
        det_colors = sum(1 for r in deterministic_results if r.get("normalized_color"))
        det_sizes = sum(1 for r in deterministic_results if r.get("normalized_size"))
        logger.info(
            "Deterministic pass: %d colors, %d sizes resolved (of %d products)",
            det_colors,
            det_sizes,
            len(products),
        )

        logger.info(
            "Gardener LLM config: model=%s, reasoning=%s",
            model_key,
            reasoning_effort,
        )

        # ── Pass 2: LLM normalization ──
        tenant_id = state.get("tenant_id", "traverse")
        execution_mode = state.get("execution_mode", "sync")

        # Handle Batch API Flow
        if execution_mode == "batch_status":
            from contextrouter.core.config import get_config
            from contextrouter.modules.models.llm.openai_batch import OpenAIBatchClient

            client = OpenAIBatchClient(get_config())
            job_id = state.get("batch_job_id", "")
            if not job_id:
                return {"errors": ["missing batch_job_id for status request"]}

            job = await client.get_batch(job_id)
            return {
                "results": [],
                "taxonomy_candidates": [],
                "batch_info": {
                    "id": job.id,
                    "status": job.status,
                    "request_counts": job.request_counts,
                },
            }

        elif execution_mode == "batch_import":
            from contextrouter.core.config import get_config
            from contextrouter.cortex.graphs.commerce.gardener.normalizer import (
                _parse_json_response,
            )
            from contextrouter.modules.models.llm.openai_batch import OpenAIBatchClient

            client = OpenAIBatchClient(get_config())
            job_id = state.get("batch_job_id", "")
            if not job_id:
                return {"errors": ["missing batch_job_id for import request"]}

            job = await client.get_batch(job_id)
            if job.status != "completed":
                return {"errors": [f"Cannot import batch {job_id} in status: {job.status}"]}

            batch_results = await client.get_batch_results(job)
            llm_results = []

            for item in batch_results:
                if item.error:
                    logger.error("Batch error %s: %s", item.custom_id, item.error)
                    continue
                parsed = _parse_json_response(item.content)
                llm_results.extend(parsed)

            # For imports, the LLM results are directly merged as final. They already have `method=llm` marked
            # We don't have deterministic results here (they were written during submit)
            results = []
            taxonomy_candidates = []

            for llm_res in llm_results:
                llm_res["method"] = "llm"  # guarantee it's marked
                results.append(llm_res)
                for candidate in llm_res.get("taxonomy_candidates", []):
                    taxonomy_candidates.append(candidate)

            return {
                "results": results,
                "taxonomy_candidates": taxonomy_candidates,
                "batch_info": {"imported": len(results)},
                "step_traces": state.get("step_traces", [])
                + [
                    {
                        "step": "normalize_import",
                        "total": len(results),
                        "duration_ms": int((time.time() - start) * 1000),
                    }
                ],
            }

        elif execution_mode == "batch_submit":
            if not model_key.startswith("openai/"):
                return {
                    "results": [],
                    "taxonomy_candidates": [],
                    "errors": [
                        f"Batch Submission requires an OpenAI model. Currently configured: {model_key}"
                    ],
                    "step_traces": [{"error": "Unsupported model for Batch API"}],
                }

            from contextrouter.core.config import get_config
            from contextrouter.cortex.graphs.commerce.gardener.normalizer import prepare_llm_payload
            from contextrouter.modules.models.llm.openai_batch import (
                BatchRequest,
                OpenAIBatchClient,
            )

            chunk_size = 50
            requests = []

            # Find products that still need LLM (didn't fully resolve in deterministic pass)
            {r["id"]: r for r in deterministic_results}
            llm_products = products  # pass all to LLM anyway, or only those missing fields? We pass all for now to extract name, gender, category

            for i in range(0, len(llm_products), chunk_size):
                chunk = llm_products[i : i + chunk_size]
                system_prompt, user_message = prepare_llm_payload(
                    chunk, taxonomy, examples, deterministic_results, custom_hint
                )
                req_id = f"gardener_{state.get('trace_id', 'unknown')}_{i}"
                requests.append(
                    {
                        "id": req_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message},
                        ],
                    }
                )

            batch_job_id = None
            if requests:
                client = OpenAIBatchClient(get_config())
                batch_requests = [
                    BatchRequest(
                        custom_id=req["id"],
                        messages=req["messages"],
                        model="gpt-4o-mini",
                        response_format={"type": "json_object"},
                    )
                    for req in requests
                ]
                brand = state.get("brand", "")
                source = state.get("source", "unknown")
                job = await client.create_batch(
                    batch_requests,
                    description=f"Gardener Normalization brand={brand} source={source}",
                )
                batch_job_id = job.id

            # Return deterministic results to be written immediately, and the batch ID for LLM
            return {
                "results": deterministic_results,
                "taxonomy_candidates": [],
                "batch_job_id": batch_job_id,
                "batch_info": {"job_id": batch_job_id, "status": "validating"}
                if batch_job_id
                else {},
                "step_traces": state.get("step_traces", [])
                + [
                    {
                        "step": "normalize_batch_submit",
                        "deterministic_colors": det_colors,
                        "deterministic_sizes": det_sizes,
                        "duration_ms": int((time.time() - start) * 1000),
                    }
                ],
            }

        else:
            # Sync execution
            llm_results = await run_llm_pass(
                products,
                taxonomy,
                examples,
                deterministic_results,
                custom_hint,
                tenant_id=tenant_id,
                model_key=model_key,
                reasoning_effort=reasoning_effort,
            )
            results = merge_results(deterministic_results, llm_results)

            # Collect taxonomy candidates
            taxonomy_candidates = []
            for result in results:
                for candidate in result.get("taxonomy_candidates", []):
                    taxonomy_candidates.append(candidate)

        step_trace = {
            "step": "normalize",
            "total": len(products),
            "deterministic_colors": det_colors,
            "deterministic_sizes": det_sizes,
            "llm_processed": len(llm_results),
            "taxonomy_candidates": len(taxonomy_candidates),
            "tokens": total_tokens,
            "duration_ms": int((time.time() - start) * 1000),
        }

        return {
            "results": results,
            "taxonomy_candidates": taxonomy_candidates,
            "total_tokens": state.get("total_tokens", 0) + total_tokens,
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return normalize


# ── Node 3: write_results ──


def make_write_results():
    """Create write_results node."""

    async def write_results(state: GardenerState) -> dict:
        """Write normalized fields back to Commerce DB via BiDi."""
        import uuid
        from datetime import datetime, timezone

        start = time.time()
        results = state.get("results", [])
        source = state.get("source", "dealer")
        custom_hint = state.get("custom_hint", "")
        trace_id = state.get("trace_id", uuid.uuid4().hex[:12])

        tenant_id = state.get("tenant_id", "traverse")

        if not results:
            return {
                "stats": {"total": 0, "written": 0},
                "errors": [],
            }

        bidi = GardenerBiDi(trace_id, tenant_id=tenant_id)
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        updates = []
        for result in results:
            method = result.get("method", "llm")
            if custom_hint:
                method = "llm_with_hint"

            update = {
                "id": result.get("id"),
                "product_type": result.get("product_type"),
                "model_name": result.get("model_name"),
                "normalized_category": result.get("normalized_category"),
                "normalized_color": result.get("normalized_color"),
                "normalized_size": result.get("normalized_size"),
                "enrichment_gardener": {
                    "version": "2.0",
                    "normalized_at": now_iso,
                    "original_color": result.get("original_color"),
                    "manufacturer_sku": result.get("manufacturer_sku"),
                    "gender": result.get("gender"),
                    "extra": result.get("extra"),
                    "method": method,
                    "confidence": 0.9 if result.get("product_type") else 0.5,
                    "custom_hint": custom_hint or None,
                    "taxonomy_candidates": result.get("taxonomy_candidates", []),
                },
            }
            updates.append(update)

        written = await bidi.update_normalized_products(updates, source=source)

        step_trace = {
            "step": "write_results",
            "source": source,
            "total": len(results),
            "written": written,
            "duration_ms": int((time.time() - start) * 1000),
        }

        stats = {
            "total": len(state.get("products", [])),
            "normalized": written,
            "deterministic": sum(1 for r in results if "deterministic" in (r.get("method") or "")),
            "llm": sum(1 for r in results if r.get("method") == "llm"),
            "taxonomy_candidates": len(state.get("taxonomy_candidates", [])),
        }

        logger.info(
            "Gardener v2 complete: %d/%d written (source=%s, brand=%s)",
            written,
            len(results),
            source,
            state.get("brand", ""),
        )

        return {
            "stats": stats,
            "errors": state.get("errors", []),
            "step_traces": state.get("step_traces", []) + [step_trace],
        }

    return write_results


# ── Helpers (used by tests) ──


def _load_prompt(prompts_dir: str, filename: str) -> str:
    """Load prompt template from file."""
    import os

    path = os.path.join(prompts_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path) as f:
        return f.read()


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug."""
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text


def _parse_json_response(content: str) -> list:
    """Parse JSON array from LLM response, handling markdown wrapping."""
    import json
    import re

    # Try direct parse
    content = content.strip()
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []
