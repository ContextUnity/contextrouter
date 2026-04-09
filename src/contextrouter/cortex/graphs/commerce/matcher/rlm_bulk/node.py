"""LangGraph node for RLM Bulk Matcher — brand-by-brand BiDi iteration."""

import time
import uuid
from typing import Any

from contextcore import get_context_unit_logger

from .bidi import BiDiClient
from .matcher import RLMBulkMatcher
from .utils import slim_products

logger = get_context_unit_logger(__name__)


async def rlm_bulk_match_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node for RLM bulk matching.

    Strategy: brand-by-brand BiDi iteration.
    1. Fetch list of matchable brands via BiDi (brands with products on both sides).
    2. For each brand:
       a) Fetch supplier products for that brand via BiDi (small payload).
       b) Fetch site products for that brand via BiDi (small payload).
       c) Run RLM matching.
       d) Upload results via BiDi.
    3. Aggregate stats.
    """
    confidence_threshold = state.get("confidence_threshold", 0.7)
    dealer_code = state.get("dealer_code", "")
    target_brand = state.get("target_brand", "")
    force_not_matched = state.get("force_not_matched", False)

    run_id = uuid.uuid4().hex[:12]
    client = BiDiClient(run_id)

    # ── Step 1: Fetch taxonomies and manual matches ──
    await client.report(
        event="stage",
        stage="init",
        status="running",
        message="Завантаження таксономій та шаблонів…",
    )

    try:
        logger.info("Fetching taxonomies via BiDi")
        taxonomies = await client.export_taxonomies()

        logger.info("Fetching manual matches via BiDi")
        manual_matches, wrong_pairs = await client.export_manual_matches()

        await client.report(
            event="stage",
            stage="init",
            status="done",
            message="Таксономії: %d, шаблони: %d, wrong: %d"
            % (
                sum(len(v) for v in taxonomies.values() if isinstance(v, list)),
                len(manual_matches),
                len(wrong_pairs),
            ),
        )
    except Exception as e:
        logger.error("Failed to fetch taxonomies/manual_matches: %s", e)
        await client.report(
            event="stage",
            stage="init",
            status="error",
            message="Помилка: %s" % str(e),
        )
        return {"matches": [], "match_stats": {"error": f"bidi_fetch_failed: {e}"}, "unmatched": []}

    # ── Step 2: Fetch Oscar brands ──
    await client.report(
        event="stage",
        stage="brands",
        status="running",
        message="Завантаження брендів Oscar…",
    )

    try:
        if target_brand:
            brands_data = [{"brand": target_brand, "site_count": 0}]
            logger.info("Single-brand mode: '%s'", target_brand)
        else:
            logger.info("Fetching Oscar brands via BiDi")
            brands_data = await client.export_matcher_brands(dealer_code)
            logger.info("Found %d Oscar brands", len(brands_data))

        await client.report(
            event="stage",
            stage="brands",
            status="done",
            message="Знайдено %d брендів" % len(brands_data),
            brands_total=len(brands_data),
        )
    except Exception as e:
        logger.error("Failed to fetch brands: %s", e)
        await client.report(
            event="stage",
            stage="brands",
            status="error",
            message="Помилка: %s" % str(e),
        )
        return {
            "matches": [],
            "match_stats": {"error": f"bidi_brands_fetch_failed: {e}"},
            "unmatched": [],
        }

    if not brands_data:
        logger.warning("No common brands")
        await client.report(
            event="stage",
            stage="brands",
            status="done",
            message="Немає спільних брендів для матчингу",
        )
        return {"matches": [], "match_stats": {"error": "no_common_brands"}, "unmatched": []}

    # ── Step 3: Iterate brand-by-brand ──
    await client.report(
        event="stage",
        stage="rlm",
        status="running",
        message="Початок RLM матчингу по брендах…",
        brands_total=len(brands_data),
    )

    # Matcher model — passed directly in dispatch payload (rlm_model field)
    # TODO: migrate matcher to template-based graph like gardener
    rlm_model = state.get("rlm_model", "") or "mercury-2"
    rlm_reasoning = state.get("rlm_reasoning", "") or "none"
    logger.info("Matcher LLM config: model=%s, reasoning=%s", rlm_model, rlm_reasoning)

    all_matches = []
    brand_stats = {}
    total_supplier = 0
    total_site = 0

    matcher = RLMBulkMatcher(
        environment="local",
        verbose=True,
        tenant_id=state.get("tenant_id", ""),
        rlm_model=rlm_model,
        reasoning_effort=rlm_reasoning,
    )

    custom_prompt = state.get("custom_prompt", "") or ""
    resume_brand_lower = None
    if custom_prompt.upper().startswith("RESUME:"):
        resume_brand_lower = custom_prompt.split(":", 1)[1].strip().lower()

    skipping_for_resume = bool(resume_brand_lower)

    for idx, brand_info in enumerate(brands_data):
        if await client.check_abort():
            logger.info("RLM Bulk Matcher aborted.")
            await client.report(
                event="stage", stage="rlm", status="error", message="Скасовано користувачем."
            )
            break

        brand_name = brand_info["brand"]
        brand_start = time.time()

        if skipping_for_resume:
            if brand_name.lower() == resume_brand_lower:
                skipping_for_resume = False
            else:
                logger.info("Resuming: skipping brand '%s'", brand_name)
                continue

        await client.report(
            event="brand",
            brand=brand_name,
            status="fetching",
            message="Завантаження продуктів…",
            brands_done=idx,
            brands_total=len(brands_data),
        )

        try:
            brand_suppliers = await client.export_unmatched_products(
                dealer_code, brand_name, force_not_matched
            )
            brand_site = await client.export_site_products(brand_name)

            total_supplier += len(brand_suppliers)
            total_site += len(brand_site)

            if not brand_suppliers or not brand_site:
                logger.info("Brand '%s': skipping", brand_name)
                brand_stats[brand_name] = {
                    "supplier": len(brand_suppliers),
                    "site": len(brand_site),
                    "matched": 0,
                    "skipped": True,
                }
                await client.report(
                    event="brand",
                    brand=brand_name,
                    status="skipped",
                    message="Пропущено",
                    suppliers=len(brand_suppliers),
                    site=len(brand_site),
                    elapsed_ms=int((time.time() - brand_start) * 1000),
                    brands_done=idx + 1,
                    brands_total=len(brands_data),
                )
                continue

            await client.report(
                event="brand",
                brand=brand_name,
                status="matching",
                message="Аналіз: %d постач. → %d сайт" % (len(brand_suppliers), len(brand_site)),
                suppliers=len(brand_suppliers),
                site=len(brand_site),
                brands_done=idx,
                brands_total=len(brands_data),
            )

            slim_suppliers = slim_products(brand_suppliers)
            slim_site = slim_products(brand_site)

            result = await matcher.match_all(
                slim_suppliers,
                slim_site,
                confidence_threshold=confidence_threshold,
                taxonomies=taxonomies,
                manual_matches=manual_matches,
                wrong_pairs=wrong_pairs,
            )
            all_matches.extend(result.matches)

            matched_count = len({m.supplier_id for m in result.matches if m.site_id})
            brand_stats[brand_name] = {
                "supplier": len(brand_suppliers),
                "site": len(brand_site),
                "matched": matched_count,
                "match_rate": result.stats.get("match_rate", 0),
            }

            brand_matches_payload = [
                {
                    "dealer_product_id": int(float(m.supplier_id)) if m.supplier_id else None,
                    "oscar_product_id": int(float(m.site_id)) if m.site_id else None,
                    "confidence": m.confidence,
                    "notes": "RLM %s: %s" % (m.match_type, getattr(m, "notes", "")),
                }
                for m in result.matches
                if m.supplier_id
            ]

            for p in result.unmatched:
                if p.get("id"):
                    brand_matches_payload.append(
                        {
                            "dealer_product_id": int(float(p["id"])),
                            "oscar_product_id": None,
                            "confidence": 0.0,
                            "notes": "RLM unmatched: AI found no match",
                        }
                    )

            if brand_matches_payload:
                try:
                    saved = await client.bulk_link_products(brand_matches_payload)
                    brand_stats[brand_name]["saved"] = saved
                except Exception as save_err:
                    logger.error("Failed to save matches: %s", save_err)
                    brand_stats[brand_name]["save_error"] = str(save_err)

            await client.report(
                event="brand",
                brand=brand_name,
                status="done",
                message="%d/%d матчів" % (matched_count, len(brand_suppliers)),
                suppliers=len(brand_suppliers),
                site=len(brand_site),
                matched=matched_count,
                elapsed_ms=int((time.time() - brand_start) * 1000),
                tokens=result.stats.get("total_tokens", 0),
                brands_done=idx + 1,
                brands_total=len(brands_data),
            )

        except Exception as e:
            logger.error("RLM matching failed for brand '%s': %s", brand_name, e)
            brand_stats[brand_name] = {"error": str(e)}
            await client.report(
                event="brand",
                brand=brand_name,
                status="error",
                message="Помилка: %s" % str(e),
                elapsed_ms=int((time.time() - brand_start) * 1000),
                brands_done=idx + 1,
                brands_total=len(brands_data),
            )

    # ── Step 4: Aggregate stats ──
    total_matched = sum(s.get("matched", 0) for s in brand_stats.values())
    overall_rate = total_matched / total_supplier if total_supplier else 0

    aggregate_stats = {
        "brands_processed": len(brands_data),
        "total_matched": total_matched,
        "total_supplier": total_supplier,
        "total_site": total_site,
        "match_rate": overall_rate,
        "brand_stats": brand_stats,
    }

    logger.info(
        "Brand-by-brand matching complete: %d/%d (%.1f%%)",
        total_matched,
        total_supplier,
        overall_rate * 100,
    )

    await client.report(
        event="stage",
        stage="rlm",
        status="done",
        message="Завершено: %d/%d матчів (%.1f%%), %d брендів"
        % (total_matched, total_supplier, overall_rate * 100, len(brands_data)),
        matched=total_matched,
        brands_done=len(brands_data),
        brands_total=len(brands_data),
    )

    return {"matches": [], "match_stats": aggregate_stats, "unmatched": []}
