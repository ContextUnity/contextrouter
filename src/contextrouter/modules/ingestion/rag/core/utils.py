"""Shared ingestion helpers.

Keep ingestion plugins thin by centralizing common, deterministic utilities here.
"""

from __future__ import annotations

import html
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from contextrouter.core import Config
from contextrouter.core.types import StructData, StructDataValue

from ..config import DEFAULT_TAXONOMY_PATH
from ..settings import RagIngestionConfig
from .types import GraphEnrichmentResult

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# TSV utilities (for robust LLM output parsing)
# ---------------------------------------------------------------------------


def parse_tsv_line(line: str) -> list[str]:
    """Split TSV line, handling both real tabs and literal '<TAB>'.

    Returns empty list if line has no tab separators.
    """
    if "\t" in line:
        return line.split("\t")
    if "<TAB>" in line:
        return line.split("<TAB>")
    return []


def llm_generate_tsv(
    *,
    core_cfg: Config,
    prompt: str,
    model: str,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    retries: int = 3,
) -> str:
    """LLM call with retry, returns raw text (for TSV parsing).

    Args:
        prompt: The prompt to send to the LLM
        model: Model identifier (e.g., "vertex/gemini-2.5-flash")
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        retries: Number of retry attempts

    Returns:
        Raw text response, or empty string if all retries fail.
    """
    import time

    from ..utils.llm import llm_generate

    for attempt in range(retries):
        try:
            raw = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                parse_json=False,
            )
            text = raw if isinstance(raw, str) else str(raw or "")
            text = text.strip()

            # Retry on empty or malformed output (just headers, no data)
            if not text:
                if attempt < retries - 1:
                    time.sleep(1 + attempt)  # Exponential backoff for rate limits
                    continue
                return ""

            # Check if output looks like valid TSV (has at least one data line after headers)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            has_entities_section = any(ln.upper().startswith("ENTITIES") for ln in lines)
            has_relations_section = any(ln.upper().startswith("RELATIONS") for ln in lines)
            # If we have sections but no data lines, retry
            if (has_entities_section or has_relations_section) and len(lines) <= 2:
                if attempt < retries - 1:
                    time.sleep(1 + attempt)
                    continue

            return text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1 + attempt)  # Exponential backoff
                continue
            logger.debug("llm_generate_tsv: all retries exhausted: %s", e)
    return ""


def load_taxonomy_safe(taxonomy_path: Path | None = None) -> StructData | None:
    """Load taxonomy JSON from disk with graceful degradation."""
    path = taxonomy_path or DEFAULT_TAXONOMY_PATH
    if not isinstance(path, Path):
        return None
    if not path.exists():
        logger.debug("Taxonomy file not found: %s", path)
        return None
    try:
        with open(path, encoding="utf-8") as f:
            obj: StructDataValue = json.load(f)
            return obj if isinstance(obj, dict) else None
    except Exception as e:
        logger.warning("Failed to load taxonomy: %s", e)
        return None


def normalize_ambiguous_unicode(text: str) -> str:
    """Normalize ambiguous unicode characters to ASCII equivalents.

    Replaces common ambiguous unicode characters that can cause display issues:
    - Em-dash (—) → --
    - En-dash (–) → -
    - Ellipsis (…) → ...
    - Smart quotes → straight quotes
    - Fullwidth ASCII variants → ASCII
    - Other common ambiguous characters

    Args:
        text: Text potentially containing ambiguous unicode

    Returns:
        Text with ambiguous unicode normalized to ASCII
    """
    # Unicode to ASCII mappings
    replacements = {
        # Dashes and punctuation
        "\u2014": "--",  # Em-dash
        "\u2013": "-",  # En-dash
        "\u2026": "...",  # Ellipsis
        # Smart quotes
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote (curly apostrophe)
        "\u201a": ",",  # Single low-9 quote
        "\u201e": '"',  # Double low-9 quote
        # Other symbols
        "\u2022": "*",  # Bullet
        "\u00a0": " ",  # Non-breaking space
        # Fullwidth ASCII variants (common in CJK contexts, YouTube titles)
        "\uff1a": ":",  # Fullwidth colon
        "\uff01": "!",  # Fullwidth exclamation
        "\uff1f": "?",  # Fullwidth question mark
        "\uff0c": ",",  # Fullwidth comma
        "\uff0e": ".",  # Fullwidth period
        "\uff1b": ";",  # Fullwidth semicolon
    }

    result = text
    for unicode_char, ascii_replacement in replacements.items():
        result = result.replace(unicode_char, ascii_replacement)

    return result


def normalize_clean_text(text: str) -> str:
    """Normalize UI text fields (quote/answer) to a clean, human-readable string."""
    # Decode HTML entities that can leak from web sources / platform metadata.
    # Example: "it&#39;s" -> "it's"
    s = html.unescape(text).replace("\n", " ").strip()
    return " ".join(s.split())


def clean_markdown_headers(text: str) -> str:
    """Clean markdown headers: strip extra spaces and normalize formatting.

    Converts:
    - "###   HEADER   " → "### HEADER"
    - "##  Chapter 1  " → "## Chapter 1"

    Args:
        text: Markdown text with potentially messy headers

    Returns:
        Cleaned markdown with normalized headers
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        # Check if line is a header
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if header_match:
            level = header_match.group(1)
            title = header_match.group(2).strip()  # Strip extra spaces from title
            cleaned_lines.append(f"{level} {title}")
        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def strip_markdown_from_text(text: str) -> str:
    """Strip markdown formatting from text (headers, bold, italic, etc.).

    Removes markdown syntax but keeps the text content:
    - "## Chapter 1" → "Chapter 1"
    - "**Bold**" → "Bold"
    - "_Italic_" → "Italic"

    Args:
        text: Text with markdown formatting

    Returns:
        Plain text without markdown syntax
    """
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove bold/italic
    text = re.sub(r"\*\*([^\*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^\*]+)\*", r"\1", text)  # *bold* or *italic*
    text = re.sub(r"\_\_([^\_]+)\_\_", r"\1", text)  # __bold__
    text = re.sub(r"\_([^\_]+)\_", r"\1", text)  # _italic_

    # Remove markdown links [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    return text.strip()


def filter_testimonial_signatures(content: str) -> str:
    """Filter out testimonial signature lines that shouldn't be treated as chapters.

    Removes or marks lines starting with --, —, or - at the start of a line
    (these are testimonial signatures in book PDFs).

    Args:
        content: Markdown content with potential testimonial signatures

    Returns:
        Content with testimonial signatures converted to regular content
    """
    lines = content.split("\n")
    filtered_lines = []

    for line in lines:
        stripped = line.strip()
        # Check if line is a testimonial signature (starts with --, —, or -)
        if (
            stripped.startswith("--")
            or stripped.startswith("—")
            or (stripped.startswith("- ") and len(stripped) < 100)
        ):
            # Convert header to regular content (remove # if present)
            if stripped.startswith("#"):
                # It's a header, remove the header marker
                content_line = re.sub(r"^#{1,6}\s*", "", stripped)
                filtered_lines.append(content_line)
            else:
                # Regular testimonial line, keep as-is
                filtered_lines.append(line)
        else:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


# ---------------------------------------------------------------------------
# Parallel utilities (bounded ThreadPoolExecutor)
# ---------------------------------------------------------------------------


def resolve_workers(*, config: RagIngestionConfig, workers: int | None) -> int:
    """Resolve requested worker count.

    - If workers is None or <= 0, uses config.ingestion.workers.
    - Always clamps to >= 1.
    """
    w = 0 if workers is None else int(workers)
    if w <= 0:
        w = int(config.ingestion.workers)
    return max(1, int(w))


def parallel_map(
    items: Iterable[T],
    fn: Callable[[T], R],
    *,
    workers: int,
    ordered: bool = False,
    swallow_exceptions: bool = True,
) -> list[R | None]:
    """Run fn(item) across items with bounded parallelism.

    Returns list of results. If ordered=True, preserves input order (None for failed tasks).
    """
    items_list = list(items)
    if not items_list:
        return []

    w = max(1, int(workers))
    if w <= 1 or len(items_list) <= 1:
        out: list[R | None] = []
        for it in items_list:
            try:
                out.append(fn(it))
            except Exception:
                if swallow_exceptions:
                    out.append(None)
                else:
                    raise
        return out

    if ordered:
        out2: list[R | None] = [None] * len(items_list)
        with ThreadPoolExecutor(max_workers=w) as ex:
            fut_to_idx = {ex.submit(fn, it): i for i, it in enumerate(items_list)}
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    out2[idx] = fut.result()
                except Exception:
                    if not swallow_exceptions:
                        raise
                    out2[idx] = None
        return out2

    out3: list[R | None] = []
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = [ex.submit(fn, it) for it in items_list]
        for fut in as_completed(futs):
            try:
                out3.append(fut.result())
            except Exception:
                if not swallow_exceptions:
                    raise
                out3.append(None)
    return out3


def build_enriched_input_text(
    *,
    content: str,
    keywords: list[str] | None = None,
    summary: str | None = None,
    parent_categories: list[str] | None = None,
) -> str:
    """Build ShadowRecord.input_text (search payload) with natural-language enrichment."""
    input_text = content

    enrichment_parts: list[str] = []

    if parent_categories:
        cats = [c for c in parent_categories if isinstance(c, str) and c.strip()]
        if cats:
            enrichment_parts.append(f"Categories: {', '.join(cats[:5])}.")

    if isinstance(summary, str) and summary.strip():
        enrichment_parts.append(f"Additional Knowledge: {summary.strip()}")

    kws = [k for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if kws:
        top = kws[:10]
        if len(top) == 1:
            enrichment_parts.append(f"This text is related to {top[0]}.")
        elif len(top) <= 3:
            enrichment_parts.append(f"This text is related to {', '.join(top[:-1])} and {top[-1]}.")
        else:
            enrichment_parts.append(
                f"This text is related to {', '.join(top[:5])}, and other concepts."
            )

    if enrichment_parts:
        input_text += "\n\n" + " ".join(enrichment_parts)

    return input_text


def get_graph_enrichment(
    *,
    text: str,
    enrichment_func: Callable[[str], GraphEnrichmentResult],
) -> tuple[list[str], str, list[str]]:
    """Normalize GraphEnricher output to (keywords, summary, parent_categories)."""
    try:
        enrichment = enrichment_func(text)
    except Exception:
        return ([], "", [])

    if not isinstance(enrichment, dict):
        return ([], "", [])

    keywords_raw = enrichment.get("keywords", [])
    keywords: list[str] = []
    if isinstance(keywords_raw, list):
        for k in keywords_raw:
            if isinstance(k, str) and k.strip():
                keywords.append(k.strip())
            if len(keywords) >= 50:
                break

    summary = enrichment.get("summary", "")
    summary_out = summary.strip() if isinstance(summary, str) else ""

    cats_raw = enrichment.get("parent_categories", [])
    cats: list[str] = []
    if isinstance(cats_raw, list):
        for c in cats_raw:
            if isinstance(c, str) and c.strip():
                cats.append(c.strip())
            if len(cats) >= 20:
                break

    return (keywords, summary_out, cats)


def clean_str_list(
    items: list[object] | None,
    *,
    limit: int,
    dedupe_case_insensitive: bool = True,
) -> list[str]:
    """Normalize a list of mixed items into a list of strings (optionally de-duped)."""
    out: list[str] = []
    seen: set[str] = set()
    for it in items or []:
        if not isinstance(it, str):
            continue
        s = it.strip()
        if not s:
            continue
        key = s.lower() if dedupe_case_insensitive else s
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max(0, int(limit)):
            break
    return out
