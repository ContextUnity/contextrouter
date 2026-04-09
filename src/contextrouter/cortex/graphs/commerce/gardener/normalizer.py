"""Normalization helpers for Gardener LLM and deterministic passes."""

import json
import re
from pathlib import Path

from contextcore import get_context_unit_logger

logger = get_context_unit_logger(__name__)


def _load_prompt(filename: str = "normalize.txt") -> str:
    """Load prompt template from gardener prompts dir."""
    path = Path(__file__).parent / "prompts" / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found: {path}")


def _parse_json_response(content: str) -> list[dict]:
    """Extract JSON array from LLM response.

    Handles:
    - Clean JSON arrays
    - Markdown-fenced code blocks (```json ... ```)
    - JSON embedded in reasoning text
    """
    if not content or not content.strip():
        logger.warning("Empty LLM response")
        return []

    # Strip markdown code fences (reasoning models often wrap in ```json...```)
    cleaned = content.strip()
    fence_match = re.search(r"```(?:json)?\s*\n?(\[.*?\])\s*\n?```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a JSON array anywhere in the response
    match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the entire content as JSON
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "results" in result:
            return result["results"]
        return [result]
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response as JSON. First 500 chars: %s", cleaned[:500])
        return []


def _build_color_index(taxonomy: dict) -> dict[str, str]:
    """Build lowercase synonym → canonical key mapping from taxonomy."""
    index: dict[str, str] = {}

    # Try extracting from the colors list (flat format from BiDi)
    colors = taxonomy.get("colors", [])
    if isinstance(colors, list):
        for color in colors:
            if isinstance(color, str):
                index[color.lower()] = color
            elif isinstance(color, dict):
                name = color.get("name", "")
                key = color.get("key", name)
                if name:
                    index[name.lower()] = key
                for syn in color.get("synonyms", []):
                    index[syn.lower()] = key

    return index


def _build_size_index(taxonomy: dict) -> dict[str, str]:
    """Build lowercase synonym → canonical size mapping from taxonomy."""
    index: dict[str, str] = {}

    sizes = taxonomy.get("sizes", [])
    if isinstance(sizes, list):
        for size in sizes:
            if isinstance(size, str):
                index[size.lower()] = size
            elif isinstance(size, dict):
                name = size.get("name", "")
                key = size.get("key", name)
                if name:
                    index[name.lower()] = key
                for syn in size.get("synonyms", []):
                    index[syn.lower()] = key

    return index


def run_deterministic_pass(
    products: list[dict],
    taxonomy: dict,
) -> list[dict]:
    """Resolve colors and sizes using taxonomy synonym matching."""
    results = []

    # Build fast synonym index from taxonomy
    color_synonyms = _build_color_index(taxonomy)
    size_synonyms = _build_size_index(taxonomy)

    for product in products:
        result = {"id": product.get("id"), "method": "deterministic"}

        # Try to resolve color
        raw_color = product.get("params_color") or product.get("params", {}).get("color") or ""
        if raw_color:
            canonical = color_synonyms.get(raw_color.lower().strip())
            if canonical:
                result["normalized_color"] = canonical
                result["original_color"] = raw_color.strip()

        # Try to resolve size
        raw_size = product.get("params_size") or product.get("params", {}).get("size") or ""
        if raw_size:
            canonical = size_synonyms.get(raw_size.lower().strip())
            if canonical:
                result["normalized_size"] = canonical
            elif raw_size.strip().upper() in ("XS", "S", "M", "L", "XL", "XXL", "3XL"):
                result["normalized_size"] = raw_size.strip().upper()

        results.append(result)

    return results


def prepare_llm_payload(
    products: list[dict],
    taxonomy: dict,
    examples: list[dict],
    deterministic_results: list[dict],
    custom_hint: str = "",
) -> tuple[str, str]:
    """Prepare system prompt and user message for Gardener LLM pass."""
    prompt_template = _load_prompt("normalize.txt")

    colors_list = taxonomy.get("colors", [])
    if isinstance(colors_list, list):
        colors_str = ", ".join(c if isinstance(c, str) else c.get("name", "") for c in colors_list)
    else:
        colors_str = str(colors_list)

    categories_list = taxonomy.get("categories", [])
    if isinstance(categories_list, list):
        cat_parts = []
        for c in categories_list:
            if isinstance(c, str):
                cat_parts.append(f'- slug="{c}"')
            elif isinstance(c, dict):
                slug = c.get("slug", "")
                name = c.get("name", "")
                if slug:
                    if name and name != slug:
                        cat_parts.append(f'- slug="{slug}" // name="{name}"')
                    else:
                        cat_parts.append(f'- slug="{slug}"')
        categories_str = "\n".join(cat_parts)
    else:
        categories_str = str(categories_list)

    sizes_list = taxonomy.get("sizes", [])
    if isinstance(sizes_list, list):
        sizes_str = ", ".join(c if isinstance(c, str) else c.get("name", "") for c in sizes_list)
    else:
        sizes_str = str(sizes_list)

    if examples:
        example_rows = []
        for ex in examples[:10]:
            name = ex.get("name", ex.get("title", ""))
            pt = ex.get("product_type", "")
            mn = ex.get("model_name", "")
            cat = ex.get("normalized_category", "")
            col = ex.get("normalized_color", "")
            oc = ex.get("original_color", "")
            sz = ex.get("normalized_size", "")
            g = ex.get("gender", "")
            example_rows.append(
                f"| {name} | product_type={pt}, model_name={mn}, "
                f"category={cat}, color={col}, original_color={oc}, "
                f"size={sz}, gender={g} |"
            )
        examples_section = (
            "## Examples of already-normalized products for this brand:\n\n"
            "| Name | Result |\n|------|--------|\n" + "\n".join(example_rows)
        )
    else:
        examples_section = ""

    hint_section = ""
    if custom_hint:
        hint_section = f"## Operator hints:\n{custom_hint}"

    system_prompt = prompt_template.format(
        categories=categories_str,
        colors=colors_str,
        sizes=sizes_str,
        examples_section=examples_section,
        hint_section=hint_section,
        product_count=len(products),
    )

    product_items = []
    det_map = {r["id"]: r for r in deterministic_results}

    for product in products:
        item = {
            "id": product.get("id"),
            "name": product.get("name", ""),
            "oscar_name": product.get("oscar_name", ""),
            "brand": product.get("brand", ""),
            "category": product.get("category", ""),
            "variant": product.get("variant", ""),
            "sku": product.get("sku", ""),
        }
        det = det_map.get(product.get("id"), {})
        if det.get("normalized_color"):
            item["hint_color"] = det["normalized_color"]
        if det.get("normalized_size"):
            item["hint_size"] = det["normalized_size"]

        product_items.append(item)

    user_message = json.dumps(product_items, ensure_ascii=False)
    return system_prompt, user_message


async def run_llm_pass(
    products: list[dict],
    taxonomy: dict,
    examples: list[dict],
    deterministic_results: list[dict],
    custom_hint: str = "",
    tenant_id: str = "traverse",
    model_key: str = "openai/gpt-5-mini",
    reasoning_effort: str = "minimal",
    api_key: str | None = None,
) -> list[dict]:
    """Run LLM normalization for product_type, model_name, etc."""
    from contextrouter.modules.models import model_registry

    system_prompt, user_message = prepare_llm_payload(
        products, taxonomy, examples, deterministic_results, custom_hint
    )

    # Create LLM — api_key from payload takes priority, otherwise Shield lookup
    create_kwargs: dict = {
        "tenant_id": tenant_id,
        "shield_key_name": "gardener_model",
    }
    if api_key:
        create_kwargs["api_key"] = api_key

    try:
        model = model_registry.create_llm(model_key, **create_kwargs)
    except Exception as e:
        logger.error("Failed to create LLM for Gardener: %s", e)
        return []

    # Call LLM
    try:
        from contextrouter.modules.models.types import ModelRequest, TextPart

        req_kwargs = {
            "system": system_prompt,
            "parts": [TextPart(text=user_message)],
            "temperature": 0.5,  # Mercury-2 min=0.5; for other models still low enough
            "max_output_tokens": 16384,
            "response_format": "json_object",
        }

        response = await model.generate(ModelRequest(**req_kwargs))
        content = response.text

        logger.info(
            "LLM raw response (%d chars, model=%s): %s",
            len(content),
            model_key,
            content[:300],
        )

        results = _parse_json_response(content)

        logger.info("LLM returned %d normalization results", len(results))
        if not results and content:
            logger.error(
                "LLM returned text but 0 parsed results. Full response:\n%s",
                content[:2000],
            )
        return results

    except Exception as e:
        logger.error("LLM normalization failed: %s", e)
        return []


def merge_results(
    deterministic: list[dict],
    llm: list[dict],
) -> list[dict]:
    """Merge deterministic color/size results with LLM extraction results."""
    det_map = {r["id"]: r for r in deterministic}
    llm_map = {r["id"]: r for r in llm}

    merged = []
    all_ids = set(det_map.keys()) | set(llm_map.keys())

    for product_id in all_ids:
        det = det_map.get(product_id, {})
        llm_result = llm_map.get(product_id, {})

        result = {"id": product_id}

        # LLM provides product_type, model_name, manufacturer_sku, gender, category, extra
        for field in (
            "product_type",
            "model_name",
            "manufacturer_sku",
            "normalized_category",
            "gender",
            "taxonomy_candidates",
            "extra",
        ):
            if field in llm_result:
                result[field] = llm_result[field]

        # Color: prefer deterministic, fallback to LLM
        result["normalized_color"] = det.get("normalized_color") or llm_result.get(
            "normalized_color"
        )
        result["original_color"] = det.get("original_color") or llm_result.get("original_color")

        # Size: prefer deterministic, fallback to LLM
        result["normalized_size"] = det.get("normalized_size") or llm_result.get("normalized_size")

        # Method tracking
        if det.get("normalized_color") or det.get("normalized_size"):
            result["method"] = "deterministic+llm" if llm_result else "deterministic"
        else:
            result["method"] = "llm" if llm_result else "none"

        merged.append(result)

    return merged
