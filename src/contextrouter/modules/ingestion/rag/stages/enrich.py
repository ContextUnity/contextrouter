"""Enrichment stage: add NER + keyphrases metadata to CleanText."""

from __future__ import annotations

import asyncio
import logging
import time

from contextrouter.core import BisquitEnvelope, Config
from contextrouter.modules.ingestion.rag.config import get_assets_paths
from contextrouter.modules.ingestion.rag.core.utils import resolve_workers
from contextrouter.modules.ingestion.rag.settings import RagIngestionConfig
from contextrouter.modules.ingestion.rag.stages.store import (
    read_raw_data_jsonl,
    write_raw_data_jsonl,
)
from contextrouter.modules.transformers.keyphrases import KeyphraseTransformer
from contextrouter.modules.transformers.ner import NERTransformer

logger = logging.getLogger(__name__)


def _merge_keywords(existing: object, new: object) -> list[str]:
    base = (
        [str(x).strip() for x in (existing or []) if str(x).strip()]
        if isinstance(existing, list)
        else []
    )
    add = [str(x).strip() for x in (new or []) if str(x).strip()] if isinstance(new, list) else []
    merged: list[str] = []
    seen: set[str] = set()
    for item in base + add:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


async def enrich_clean_text(
    *,
    config: RagIngestionConfig,
    core_cfg: Config,
    only_types: list[str],
    overwrite: bool = True,
    workers: int = 1,
) -> dict[str, str]:
    if not config.enrichment.ner_enabled and not config.enrichment.keyphrases_enabled:
        logger.info("enrich: disabled (enrichment.ner_enabled/keyphrases_enabled=false)")
        return {}

    if not overwrite:
        logger.warning("enrich: overwrite=false, skipping (would rewrite clean_text files)")
        return {}

    paths = get_assets_paths(config)
    out_paths: dict[str, str] = {}

    async def _enrich_items(items):
        ner = None
        if config.enrichment.ner_enabled:
            ner = NERTransformer()
            ner.configure(
                {
                    "mode": config.enrichment.ner.mode,
                    "model": config.enrichment.ner.model,
                    "entity_types": config.enrichment.ner.entity_types,
                    "min_confidence": config.enrichment.ner.min_confidence,
                    "core_cfg": core_cfg,
                }
            )
        keyphrases = None
        if config.enrichment.keyphrases_enabled:
            keyphrases = KeyphraseTransformer()
            keyphrases.configure(
                {
                    "mode": config.enrichment.keyphrases.mode,
                    "max_phrases": config.enrichment.keyphrases.max_phrases,
                    "min_score": config.enrichment.keyphrases.min_score,
                    "model": config.enrichment.keyphrases.model,
                    "core_cfg": core_cfg,
                }
            )

        enriched = []
        for item in items:
            envelope = BisquitEnvelope(content=item.content, metadata=dict(item.metadata or {}))
            try:
                if ner:
                    envelope = await ner.transform(envelope)
                if keyphrases:
                    envelope = await keyphrases.transform(envelope)
            except Exception:
                logger.exception("enrich: failed to enrich item, keeping original metadata")
            metadata = dict(envelope.metadata or {})
            ner_entities = metadata.get("ner_entities")
            if isinstance(ner_entities, list):
                ner_texts = [
                    str(ent.get("text", "")).strip()
                    for ent in ner_entities
                    if isinstance(ent, dict)
                ]
                metadata["keywords"] = _merge_keywords(metadata.get("keywords"), ner_texts)
            if keyphrases:
                metadata["keywords"] = _merge_keywords(
                    metadata.get("keywords"), metadata.get("keyphrase_texts")
                )
            item.metadata = metadata
            enriched.append(item)
        return enriched

    async def _run_one(t: str, sem: asyncio.Semaphore) -> tuple[str, str]:
        async with sem:
            t0 = time.perf_counter()
            in_path = paths["clean_text"] / f"{t}.jsonl"

            # Blocking IO -> Thread
            try:
                items = await asyncio.to_thread(read_raw_data_jsonl, in_path)
            except Exception:
                logger.warning("enrich: failed to read %s", in_path)
                return (t, "")

            if not items:
                logger.warning("enrich: no clean_text items for type=%s at %s", t, in_path)
                return (t, "")

            # Async Logic
            enriched = await _enrich_items(items)

            out_path = paths["clean_text"] / f"{t}.jsonl"

            # Blocking IO -> Thread
            count = await asyncio.to_thread(
                write_raw_data_jsonl, enriched, out_path, overwrite=True
            )

            logger.warning(
                "enrich: wrote %d records for type=%s -> %s (%.1fs)",
                count,
                t,
                out_path,
                time.perf_counter() - t0,
            )
            return (t, str(out_path))

    w = resolve_workers(config=config, workers=workers)
    sem = asyncio.Semaphore(w)

    tasks = [_run_one(t, sem) for t in only_types]
    results = await asyncio.gather(*tasks)

    for r in results:
        if not r:
            continue
        tt, out_path = r
        if out_path:
            out_paths[tt] = out_path

    return out_paths


__all__ = ["enrich_clean_text"]
