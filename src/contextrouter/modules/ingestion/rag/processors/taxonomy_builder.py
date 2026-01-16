"""Taxonomy builder (CleanText -> taxonomy.json).

Design:
- Extract domain terms from samples using LLM
- Assign terms to predefined categories (from config) or let LLM create categories
- Predefined categories avoid mega-category problem
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from contextrouter.core import Config

from ..core.utils import (
    llm_generate_tsv,
    normalize_ambiguous_unicode,
    parallel_map,
    parse_tsv_line,
    strip_markdown_from_text,
)
from ..settings import RagIngestionConfig
from .taxonomy.sampling import (
    collect_clean_text_samples_from_dir,
)

logger = logging.getLogger(__name__)

_META_TERM_RE = re.compile(
    r"\b(public domain|original text|author'?s|preface|foreword|appendix|"
    r"toc|table of contents|chapter|page|copyright|isbn)\b",
    re.IGNORECASE,
)
_PROMO_TERM_RE = re.compile(
    r"\b(amazon|kindle|audible|youtube|subscribe|review|podcast)\b", re.IGNORECASE
)
_CONCEPT_PREFIX_RE = re.compile(r"^concepts\[\d+\](?:term)?:?\s*", re.IGNORECASE)


def build_taxonomy(
    source_root: Path,
    output_path: Path,
    config: RagIngestionConfig,
    core_cfg: Config,
    *,
    force_rebuild: bool = False,
    workers: int = 4,
) -> dict[str, Any]:
    """Build taxonomy from CleanText samples."""
    existing: dict[str, Any] | None = None
    if output_path.exists() and not force_rebuild:
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to load existing taxonomy: %s", e)

    focus = config.taxonomy.philosophy_focus.strip()
    max_samples = config.taxonomy.max_samples
    scan_model = (
        config.taxonomy.scan_model.strip() or core_cfg.models.ingestion.taxonomy.model.strip()
    )

    # Load predefined categories from config
    predefined_cats = _load_predefined_categories(config.taxonomy.categories)

    samples = collect_clean_text_samples_from_dir(
        clean_text_dir=source_root, config=config, max_samples=max_samples
    )
    logger.info(
        "taxonomy: samples=%d scan_model=%s predefined_categories=%d",
        len(samples),
        scan_model,
        len(predefined_cats),
    )

    if not samples:
        return existing or _empty_taxonomy(focus)

    # Extract terms
    terms = _extract_terms(
        core_cfg=core_cfg,
        samples=[s.text for s in samples],
        focus=focus,
        model=scan_model,
        workers=workers,
    )
    if not terms:
        logger.warning("taxonomy: no terms extracted")
        return existing or _empty_taxonomy(focus)

    # Assign terms to categories
    if predefined_cats:
        assigned = _assign_terms_to_categories(core_cfg, terms, predefined_cats, model=scan_model)
    else:
        # Fallback: use LLM-assigned categories from extraction
        assigned = terms

    # Build taxonomy structure
    new_tax = _build_taxonomy_structure(assigned, predefined_cats, focus)

    if existing:
        new_tax = _merge_taxonomy(existing, new_tax)

    new_tax = _finalize_taxonomy(new_tax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(new_tax, ensure_ascii=False, indent=2), encoding="utf-8")
    return new_tax


def _load_predefined_categories(cats: dict[str, str]) -> dict[str, str]:
    """Load predefined categories from config. Returns {name: description}."""
    return {
        _to_snake_case(k): str(v).strip()
        for k, v in cats.items()
        if isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip()
    }


def _extract_terms(
    *,
    core_cfg: Config,
    samples: list[str],
    focus: str,
    model: str,
    workers: int,
) -> list[dict[str, Any]]:
    """Extract domain terms from samples. No category assignment here."""
    batch_size = 10
    terms: list[dict[str, Any]] = []

    prompts: list[str] = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        combined = "\n\n---\n\n".join(batch)[:50000]

        prompt = f"""Extract domain concepts and terminology.

FOCUS: {focus}

CONTENT:
{combined}

Return TSV: term<TAB>synonyms<TAB>description
- Extract 15-25 high-signal concepts per batch
- term: 2-5 word phrase (domain-specific)
- synonyms: semicolon-separated (can be empty)
- description: one sentence
- No generic words, no proper names, no markdown
"""
        prompts.append(prompt)

    logger.info("taxonomy: extracting terms from %d batches", len(prompts))

    # temperature=0 for deterministic output
    if workers <= 1:
        for prompt in prompts:
            raw = llm_generate_tsv(
                core_cfg=core_cfg,
                prompt=prompt,
                model=model,
                temperature=0.0,
                max_tokens=4096,
                retries=3,
            )
            terms.extend(_parse_terms_tsv(raw))
    else:
        raws = parallel_map(
            prompts,
            lambda p: llm_generate_tsv(
                core_cfg=core_cfg,
                prompt=p,
                model=model,
                temperature=0.0,
                max_tokens=4096,
                retries=3,
            ),
            workers=workers,
            ordered=True,
            swallow_exceptions=True,
        )
        for raw in raws:
            terms.extend(_parse_terms_tsv(str(raw or "")))

    logger.info("taxonomy: extracted %d terms", len(terms))
    return terms


def _parse_terms_tsv(text: str) -> list[dict[str, Any]]:
    """Parse TSV output into term dicts."""
    out: list[dict[str, Any]] = []
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if not ln or ln.lower().startswith("term\t"):
            continue
        ln = _CONCEPT_PREFIX_RE.sub("", ln)

        parts = parse_tsv_line(ln)
        if len(parts) < 2:
            continue

        term = strip_markdown_from_text(normalize_ambiguous_unicode(parts[0].strip()))
        synonyms_s = normalize_ambiguous_unicode(parts[1].strip()) if len(parts) >= 2 else ""
        desc = normalize_ambiguous_unicode(parts[2].strip()) if len(parts) >= 3 else ""

        if not term or len(term) < 3 or len(term) > 80:
            continue
        # Normalize for word-boundary matching: treat underscores/punct as separators.
        term_lc = term.lower()
        term_lc_words = re.sub(r"[^a-z0-9]+", " ", term_lc).strip()

        # Reject obvious promo/meta/junk.
        if synonyms_s.strip().lower() in {"promo", "promotional", "advertisement", "marketing"}:
            continue
        if term.isupper() and len(term) <= 8:
            continue
        if _META_TERM_RE.search(term_lc_words) or _PROMO_TERM_RE.search(term_lc_words):
            continue
        if term.lower().startswith("concepts["):
            continue

        synonyms = [
            s.strip()
            for s in synonyms_s.split(";")
            if s.strip() and s.strip().lower() != term.lower()
        ]

        out.append(
            {"term": term, "synonyms": synonyms, "description": desc, "category": "concepts"}
        )
    return out


# ---------------------------------------------------------------------------
# Helper aliases
# ---------------------------------------------------------------------------


def _collect_clean_text_samples_from_dir(
    *, clean_text_dir: Path, config: RagIngestionConfig, max_samples: int
):
    return collect_clean_text_samples_from_dir(
        clean_text_dir=clean_text_dir, config=config, max_samples=max_samples
    )


def _parse_concepts_tsv(text: str) -> list[dict[str, Any]]:
    # Current implementation parses "terms" but outputs taxonomy concept entries.
    return _parse_terms_tsv(text)


def _assign_terms_to_categories(
    core_cfg: Config,
    terms: list[dict[str, Any]],
    categories: dict[str, str],
    *,
    model: str,
    batch_size: int = 20,
) -> list[dict[str, Any]]:
    """Assign each term to one of the predefined categories."""
    if not terms or not categories:
        return terms

    cat_names = list(categories.keys())
    cat_lines = "\n".join([f"- {name}: {desc}" for name, desc in categories.items()])

    term_list = [t["term"] for t in terms]
    term_map = {t["term"].lower(): t for t in terms}

    logger.info("taxonomy: assigning %d terms to %d categories", len(terms), len(categories))

    for start in range(0, len(term_list), batch_size):
        batch = term_list[start : start + batch_size]

        prompt = f"""Assign each term to exactly ONE category.

CATEGORIES:
{cat_lines}

TERMS:
{json.dumps(batch, ensure_ascii=False)}

Return TSV: term<TAB>category_snake_case
One line per term. Category must be one of: {", ".join(cat_names)}
"""
        raw = llm_generate_tsv(
            core_cfg=core_cfg,
            prompt=prompt,
            model=model,
            temperature=0.0,
            max_tokens=2048,
            retries=2,
        )

        for ln in (raw or "").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            parts = parse_tsv_line(ln)
            if len(parts) < 2:
                continue
            term_raw = parts[0].strip()
            cat_raw = _to_snake_case(parts[1].strip())

            if cat_raw not in cat_names:
                continue

            t = term_map.get(term_raw.lower())
            if t:
                t["category"] = cat_raw

    # Ensure all terms have a valid category
    default_cat = cat_names[0] if cat_names else "concepts"
    for t in terms:
        if t.get("category") not in cat_names:
            t["category"] = default_cat

    return terms


def _build_taxonomy_structure(
    terms: list[dict[str, Any]],
    predefined_cats: dict[str, str],
    focus: str,
) -> dict[str, Any]:
    """Build taxonomy dict from assigned terms."""
    cats: dict[str, dict[str, Any]] = {}

    # Initialize predefined categories
    for name, desc in predefined_cats.items():
        cats[name] = {"description": desc, "keywords": [], "synonyms": {}}

    # Add terms to categories
    for t in terms:
        term = t.get("term", "").strip()
        cat = t.get("category", "concepts")
        if not term:
            continue

        if cat not in cats:
            cats[cat] = {
                "description": f"Concepts related to {cat.replace('_', ' ')}",
                "keywords": [],
                "synonyms": {},
            }

        cats[cat]["keywords"].append(term)

        syns = t.get("synonyms", [])
        if syns:
            cats[cat]["synonyms"][term] = syns

    # Dedup keywords
    for cat_data in cats.values():
        seen: set[str] = set()
        uniq: list[str] = []
        for kw in cat_data.get("keywords", []):
            k = kw.lower()
            if k not in seen:
                seen.add(k)
                uniq.append(kw)
        cat_data["keywords"] = uniq

    # Remove empty categories
    cats = {k: v for k, v in cats.items() if v.get("keywords")}

    return {"philosophy_focus": focus, "categories": cats}


def _merge_taxonomy(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge new taxonomy into existing."""
    merged = {
        "philosophy_focus": new.get("philosophy_focus", existing.get("philosophy_focus", "")),
        "categories": {},
    }

    all_cats = set(existing.get("categories", {}).keys()) | set(new.get("categories", {}).keys())
    for cat in all_cats:
        e = existing.get("categories", {}).get(cat, {}) or {}
        n = new.get("categories", {}).get(cat, {}) or {}

        # Merge keywords (new first, then existing)
        seen: set[str] = set()
        kws: list[str] = []
        for kw in list(n.get("keywords", [])) + list(e.get("keywords", [])):
            if isinstance(kw, str) and kw.strip():
                k = kw.strip().lower()
                if k not in seen:
                    seen.add(k)
                    kws.append(kw.strip())

        # Merge synonyms
        syns = dict(e.get("synonyms", {}))
        syns.update(n.get("synonyms", {}))

        merged["categories"][cat] = {
            "description": n.get("description") or e.get("description") or "",
            "keywords": kws,
            "synonyms": syns,
        }

    return merged


def _finalize_taxonomy(taxonomy: dict[str, Any]) -> dict[str, Any]:
    """Add all_keywords, canonical_map, total_count."""
    all_keywords: list[str] = []
    canonical_map: dict[str, str] = {}

    for cat_data in taxonomy.get("categories", {}).values():
        if not isinstance(cat_data, dict):
            continue
        for kw in cat_data.get("keywords", []):
            if isinstance(kw, str) and kw.strip():
                all_keywords.append(kw.strip())
        for canonical, syn_list in cat_data.get("synonyms", {}).items():
            if isinstance(canonical, str) and canonical.strip():
                canonical_map[canonical.strip().lower()] = canonical.strip()
                for s in syn_list if isinstance(syn_list, list) else []:
                    if isinstance(s, str) and s.strip():
                        canonical_map[s.strip().lower()] = canonical.strip()

    # Dedup
    seen: set[str] = set()
    uniq: list[str] = []
    for kw in all_keywords:
        k = kw.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(kw)

    taxonomy["all_keywords"] = uniq
    taxonomy["canonical_map"] = canonical_map
    taxonomy["total_count"] = len(uniq)
    return taxonomy


def _empty_taxonomy(focus: str) -> dict[str, Any]:
    return {
        "philosophy_focus": focus,
        "categories": {},
        "all_keywords": [],
        "canonical_map": {},
        "total_count": 0,
    }


def _to_snake_case(val: str) -> str:
    s = (val or "").strip()
    if not s:
        return ""
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-zA-Z0-9_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s.lower().strip("_")


def _int_or(val: Any, default: int) -> int:
    try:
        return int(val)
    except Exception:
        return default
