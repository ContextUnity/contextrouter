from __future__ import annotations

import json
import logging
from pathlib import Path

from contextrouter.core.types import StructData, StructDataValue

from ...settings import RagIngestionConfig
from .common import (
    DEDUP_PREFIX_CHARS,
    TaxonomySample,
    clean_for_taxonomy_sample,
    stable_hash_u64,
    windowed_snippets,
)

logger = logging.getLogger(__name__)


def collect_clean_text_samples_from_dir(
    *, clean_text_dir: Path, config: RagIngestionConfig, max_samples: int
) -> list[TaxonomySample]:
    """Pass A: deterministic, doc-aware sampling from CleanText JSONL files."""
    include_types = config.taxonomy.include_types
    if not include_types:
        include_types = ["video", "book", "qa", "knowledge", "web"]

    paths = [clean_text_dir / f"{t}.jsonl" for t in include_types if isinstance(t, str)]
    paths = [p for p in paths if p.exists()]
    if not paths:
        return []

    # Note: sample_window_chars, sample_windows_per_item, per_type_pool are not in RagIngestionConfig yet
    # Using defaults for now - these can be added to TaxonomySection if needed
    window_chars = 1200
    windows_per_item = 9
    per_type_pool = max(200, max_samples * 2)
    per_type_pool = max(50, min(per_type_pool, 20000))

    def _doc_key(t: str, meta: StructData, content: str) -> str:
        if t == "book":
            v = meta.get("book_title") or meta.get("title") or meta.get("id")
            if isinstance(v, str) and v.strip():
                return v.strip()
        if t == "video":
            v = meta.get("video_id") or meta.get("id") or meta.get("video_url")
            if isinstance(v, str) and v.strip():
                return v.strip()
        if t == "web":
            v = meta.get("url") or meta.get("id")
            if isinstance(v, str) and v.strip():
                return v.strip()
        if t == "qa":
            v = meta.get("session_title") or meta.get("id")
            if isinstance(v, str) and v.strip():
                return v.strip()
        if t == "knowledge":
            v = meta.get("filename") or meta.get("title") or meta.get("id")
            if isinstance(v, str) and v.strip():
                return v.strip()
        return f"anon:{stable_hash_u64((content or '')[:400])}"

    type_to_samples: dict[str, list[TaxonomySample]] = {}
    for p in paths:
        t = p.stem
        counts: dict[str, int] = {}
        first: dict[str, str] = {}
        last: dict[str, str] = {}
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj: StructDataValue = json.loads(line)
                    except Exception as e:
                        logger.debug("Failed to parse JSON line: %s", e)
                        continue
                    if not isinstance(obj, dict):
                        continue
                    obj_dict: StructData = {k: v for k, v in obj.items() if isinstance(k, str)}
                    content = obj_dict.get("content")
                    if not isinstance(content, str) or not content.strip():
                        continue
                    meta_raw = obj_dict.get("metadata") or {}
                    meta: StructData = (
                        {k: v for k, v in meta_raw.items() if isinstance(k, str)}
                        if isinstance(meta_raw, dict)
                        else {}
                    )
                    dk = _doc_key(t, meta, content)
                    counts[dk] = counts.get(dk, 0) + 1
                    if dk not in first:
                        first[dk] = content
                    last[dk] = content
        except Exception as e:
            logger.warning("taxonomy: failed to read %s: %s", p, e)
            continue

        if not counts:
            continue

        mid_target = {dk: (cnt // 2) for dk, cnt in counts.items()}
        mid: dict[str, str] = {}
        seen_idx: dict[str, int] = {dk: 0 for dk in counts}
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj2: StructDataValue = json.loads(line)
                    except Exception as e:
                        logger.debug("Failed to parse JSON line: %s", e)
                        continue
                    if not isinstance(obj2, dict):
                        continue
                    obj2_dict: StructData = {k: v for k, v in obj2.items() if isinstance(k, str)}
                    content2 = obj2_dict.get("content")
                    if not isinstance(content2, str) or not content2.strip():
                        continue
                    meta2_raw = obj2_dict.get("metadata") or {}
                    meta2: StructData = (
                        {k: v for k, v in meta2_raw.items() if isinstance(k, str)}
                        if isinstance(meta2_raw, dict)
                        else {}
                    )
                    dk2 = _doc_key(t, meta2, content2)
                    i = seen_idx.get(dk2, 0)
                    if dk2 not in mid and i == mid_target.get(dk2, -1):
                        mid[dk2] = content2
                    seen_idx[dk2] = i + 1
        except Exception:
            mid = {}

        doc_best: list[tuple[int, str, str]] = []
        doc_others: list[tuple[int, str, str]] = []
        for dk in counts:
            segments: list[str] = []
            for raw_seg in [first.get(dk, ""), mid.get(dk, ""), last.get(dk, "")]:
                if not isinstance(raw_seg, str) or not raw_seg.strip():
                    continue
                s = clean_for_taxonomy_sample(raw_seg)
                if s:
                    segments.append(s)
            seg_seen: set[str] = set()
            segs_uniq: list[str] = []
            for s in segments:
                k = s[:DEDUP_PREFIX_CHARS].lower()
                if k in seg_seen:
                    continue
                seg_seen.add(k)
                segs_uniq.append(s)
            if not segs_uniq:
                continue

            snips: list[str] = []
            for s in segs_uniq:
                snips.extend(
                    windowed_snippets(s, window_chars=window_chars, max_windows=windows_per_item)
                )
            sn_seen: set[str] = set()
            sn_uniq: list[str] = []
            for sn in snips:
                if not sn:
                    continue
                k = sn[:DEDUP_PREFIX_CHARS].lower()
                if k in sn_seen:
                    continue
                sn_seen.add(k)
                sn_uniq.append(sn)
            if not sn_uniq:
                continue
            sn_uniq.sort(key=stable_hash_u64)
            best = sn_uniq[0]
            doc_best.append((stable_hash_u64(best), dk, best))
            for extra in sn_uniq[1:]:
                doc_others.append((stable_hash_u64(extra), dk, extra))

        doc_best.sort(key=lambda x: x[0])
        doc_others.sort(key=lambda x: x[0])

        chosen: list[TaxonomySample] = []
        used_prefixes: set[str] = set()

        def _maybe_add(dk: str, snippet: str) -> None:
            k = snippet[:DEDUP_PREFIX_CHARS].lower()
            if k in used_prefixes:
                return
            used_prefixes.add(k)
            chosen.append(TaxonomySample(text=snippet, source_type=t, doc_key=dk))

        for _, dk, sn in doc_best:
            if len(chosen) >= per_type_pool:
                break
            _maybe_add(dk, sn)
        for _, dk, sn in doc_others:
            if len(chosen) >= per_type_pool:
                break
            _maybe_add(dk, sn)

        chosen.sort(key=lambda s: stable_hash_u64(s.text))
        type_to_samples[t] = chosen

    if not type_to_samples:
        return []

    out: list[TaxonomySample] = []
    seen_global: set[str] = set()
    keys = [p.stem for p in paths if p.stem in type_to_samples]
    idx = 0
    while len(out) < max_samples:
        active = False
        for _ in range(len(keys)):
            t = keys[idx % len(keys)]
            idx += 1
            lst = type_to_samples.get(t) or []
            if not lst:
                continue
            active = True
            s = lst.pop(0)
            k = s.text[:DEDUP_PREFIX_CHARS].lower()
            if k in seen_global:
                continue
            seen_global.add(k)
            out.append(s)
            if len(out) >= max_samples:
                break
        if not active:
            break

    return out
