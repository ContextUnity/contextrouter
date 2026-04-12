"""Response parsing and deduplication logic for RLM bulk matcher."""

from __future__ import annotations

import ast
import json
import os
import re

from contextunity.core import get_contextunit_logger

from .types import ProductMatch

logger = get_contextunit_logger(__name__)


def parse_rlm_response(response_text: str, out_path: str = "") -> list[ProductMatch]:
    """Parse RLM response into ProductMatch objects.

    RLM executes code in its REPL and writes results to out_path.
    We read from there first, then fall back to extracting JSON from the response.
    """
    matches = []
    try:
        data = None

        # Primary: read from the file RLM wrote via json.dump(out_path)
        if out_path and os.path.exists(out_path):
            try:
                with open(out_path, "r") as f:
                    data = json.load(f)
                logger.info(
                    "Parsed %d items from RLM output file",
                    len(data) if isinstance(data, list) else 0,
                )
            except Exception as e:
                logger.warning("Failed to read RLM out_path JSON: %s", e)

        if not data:
            # Fallback: extract JSON array from response text
            json_match = re.search(r"\[[\s\S]*\]", response_text)
            if not json_match:
                return matches

            extracted = json_match.group()
            try:
                # literal_eval elegantly handles trailing commas and single quotes
                data = ast.literal_eval(extracted)
            except (ValueError, SyntaxError):
                # Fallback to JSON loads just in case
                try:
                    data = json.loads(extracted)
                except json.JSONDecodeError:
                    # Sometimes single quotes are used
                    extracted = extracted.replace("'", '"')
                    data = json.loads(extracted)

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    logger.warning("Skipping non-dict item in parsed data: %s", item)
                    continue
                # Robust confidence parsing
                raw_conf = item.get("confidence", 0)
                try:
                    conf = (
                        float(raw_conf)
                        if raw_conf is not None and str(raw_conf).lower() != "none"
                        else 0.0
                    )
                except (ValueError, TypeError):
                    conf = 0.0
                # Notes may be dict or string
                raw_notes = item.get("notes", "")
                notes_str = (
                    json.dumps(raw_notes, ensure_ascii=False)
                    if isinstance(raw_notes, dict)
                    else str(raw_notes or "")
                )
                matches.append(
                    ProductMatch(
                        supplier_id=str(item.get("supplier_id", "")),
                        supplier_name=item.get("supplier_name", ""),
                        site_id=item.get("site_id"),
                        site_name=item.get("site_name"),
                        confidence=conf,
                        match_type=item.get("match_type", "unknown"),
                        factors_matched=item.get("factors_matched", []),
                        factors_mismatched=item.get("factors_mismatched", []),
                        notes=notes_str,
                    )
                )
    except Exception as e:
        logger.error("Failed to parse RLM response: %s", e)
        logger.debug("Response was: %s", response_text[:500])

    return matches


def deduplicate_matches(
    matches: list[ProductMatch],
) -> tuple[list[ProductMatch], list[ProductMatch]]:
    """Deduplicate matches: enforce strict 1:1 matching.

    Returns:
        tuple[list[ProductMatch], list[ProductMatch]]: (final_matched, unmatched_list)
    """
    # Pass 1: each supplier → at most ONE site (keep highest confidence)
    best_by_supplier: dict[str, ProductMatch] = {}
    unmatched_list: list[ProductMatch] = []

    for m in matches:
        if not m.site_id:
            unmatched_list.append(m)
            continue

        sup_key = str(m.supplier_id)
        if sup_key not in best_by_supplier or m.confidence > best_by_supplier[sup_key].confidence:
            if sup_key in best_by_supplier:
                unmatched_list.append(
                    ProductMatch(
                        supplier_id=best_by_supplier[sup_key].supplier_id,
                        supplier_name=best_by_supplier[sup_key].supplier_name,
                        site_id=None,
                        site_name=None,
                        confidence=0.0,
                        match_type="duplicate_demoted",
                    )
                )
            best_by_supplier[sup_key] = m
        else:
            unmatched_list.append(
                ProductMatch(
                    supplier_id=m.supplier_id,
                    supplier_name=m.supplier_name,
                    site_id=None,
                    site_name=None,
                    confidence=0.0,
                    match_type="duplicate_demoted",
                )
            )

    # Pass 2: each site → at most ONE supplier (keep highest confidence)
    best_by_site: dict[str, ProductMatch] = {}
    for m in best_by_supplier.values():
        site_key = str(m.site_id)
        if site_key not in best_by_site or m.confidence > best_by_site[site_key].confidence:
            if site_key in best_by_site:
                unmatched_list.append(
                    ProductMatch(
                        supplier_id=best_by_site[site_key].supplier_id,
                        supplier_name=best_by_site[site_key].supplier_name,
                        site_id=None,
                        site_name=None,
                        confidence=0.0,
                        match_type="duplicate_demoted",
                    )
                )
            best_by_site[site_key] = m
        else:
            unmatched_list.append(
                ProductMatch(
                    supplier_id=m.supplier_id,
                    supplier_name=m.supplier_name,
                    site_id=None,
                    site_name=None,
                    confidence=0.0,
                    match_type="duplicate_demoted",
                )
            )

    final_matched = list(best_by_site.values())
    raw_matched = len([m for m in matches if m.site_id])
    if len(final_matched) < raw_matched:
        logger.info(
            "Dedup: %d raw → %d unique 1:1 matches (removed %d duplicates)",
            raw_matched,
            len(final_matched),
            raw_matched - len(final_matched),
        )

    return final_matched, unmatched_list
