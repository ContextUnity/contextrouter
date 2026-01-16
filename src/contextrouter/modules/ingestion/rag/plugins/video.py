"""Video ingestion plugin."""

from __future__ import annotations

import html
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from contextrouter.core import Config

from ..core import (
    IngestionMetadata,
    IngestionPlugin,
    RawData,
    ShadowRecord,
    register_plugin,
)
from ..core.types import GraphEnrichmentResult, VideoStructData
from ..core.utils import (
    clean_str_list,
    get_graph_enrichment,
    llm_generate_tsv,
    normalize_clean_text,
    parse_tsv_line,
)
from ..settings import RagIngestionConfig
from ..utils.records import (
    format_timestamp,
    generate_id,
)

logger = logging.getLogger(__name__)


def extract_youtube_id_from_filename(filename: str) -> tuple[str | None, str]:
    """Extract YouTube ID from filename and return clean title.

    Handles filenames like:
    - "Video Title [ABC123xyz].json" -> (ID: "ABC123xyz", Title: "Video Title")
    - "Video Title.json" -> (ID: None, Title: "Video Title")

    Args:
        filename: Filename (stem, without extension)

    Returns:
        Tuple of (youtube_id or None, cleaned_title)
    """
    # Pattern: [...] at the end, contains alphanumeric (YouTube IDs are 11 chars)
    # YouTube IDs are typically 11 characters, but we'll match any reasonable length
    pattern = r"\s*\[([A-Za-z0-9_-]{8,})\]\s*$"
    match = re.search(pattern, filename)

    if match:
        youtube_id = match.group(1)
        clean_title = filename[: match.start()].strip()
        return youtube_id, clean_title

    return None, filename.strip()


def smart_glue_words(words: list[dict]) -> list[dict]:
    """Combine words into sentences using smart gluing logic.

    Breaks on:
    - Pause > 0.8s between words
    - Word ends with punctuation (.?!)
    """
    sentences = []
    current: list[dict] = []
    start_time = 0.0

    for i, w in enumerate(words):
        word = w.get("word", "")
        word_start = w.get("start", 0.0)
        word_end = w.get("end", word_start)

        if not current:
            start_time = word_start

        # Check for pause > 0.8s
        if current:
            prev_end = current[-1].get("end", word_start)
            pause = word_start - prev_end
            if pause > 0.8:
                # Break sentence
                text = " ".join(item.get("word", "") for item in current)
                sentences.append(
                    {
                        "text": text,
                        "start": start_time,
                        "end": prev_end,
                    }
                )
                current = []
                start_time = word_start

        current.append(w)

        # Check for punctuation
        if str(word).rstrip().endswith((".", "!", "?", "...")):
            text = " ".join(item.get("word", "") for item in current)
            sentences.append(
                {
                    "text": text,
                    "start": start_time,
                    "end": word_end,
                }
            )
            current = []

    # Handle remaining words
    if current:
        text = " ".join(item.get("word", "") for item in current)
        last_end = current[-1].get("end", start_time)
        sentences.append(
            {
                "text": text,
                "start": start_time,
                "end": last_end,
            }
        )

    return sentences


def _generate_video_summaries(
    records: list[ShadowRecord],
    *,
    core_cfg: Config,
    batch_size: int,
    max_sentences: int,
    persona_name: str = "",
) -> None:
    """Generate LLM summaries for video chunks in batches.

    Updates struct_data["summary"] in-place for each record.
    Retries batches if success rate < 80%.
    After initial pass, collects failed records and retries them in smaller batches.
    """
    if not records:
        return

    import time

    # Use persona name or generic "The speaker"
    speaker_ref = persona_name if persona_name else "The speaker"

    def _process_batch(batch: list[ShadowRecord], batch_label: str = "") -> list[ShadowRecord]:
        """Process a single batch and return records that failed to get summaries.

        Args:
            batch: List of ShadowRecord to process
            batch_label: Optional label for logging (e.g., "retry batch 5")

        Returns:
            List of records that still don't have summaries
        """
        if not batch:
            return []

        batch_data = []
        for i, rec in enumerate(batch):
            quote = rec.struct_data.get("quote", "")
            video_name = rec.struct_data.get("video_name", "Video")
            batch_data.append({"idx": i, "video": video_name, "text": quote})

        prompt = f"""Create concise summaries for video transcript segments from {speaker_ref}'s live presentations.

For each segment, write 1-{max_sentences} sentences that:
- Focus on {speaker_ref}'s personal experience, insights, or teachings
- Capture the key idea, lesson, or message {speaker_ref} is sharing
- Reference {speaker_ref}'s journey, stories, or real-life examples when relevant
- Are self-contained and actionable
- Use third person (e.g., "{speaker_ref} explains...", "{speaker_ref}'s experience shows...")
- Avoid filler like "In this segment..." or "The speaker discusses..."

Return as TSV, one line per segment:
index<TAB>summary

Rules:
- Use REAL tab characters (\\t), not "<TAB>"
- Maintain order (0, 1, 2, ...)

SEGMENTS:
{json.dumps(batch_data, ensure_ascii=False)}
"""

        # Retry batch if success rate < 80%
        max_batch_retries = 3
        success_rate_threshold = 0.8

        for attempt in range(max_batch_retries):
            text = llm_generate_tsv(
                core_cfg=core_cfg,
                prompt=prompt,
                model=core_cfg.models.ingestion.preprocess.model,
                max_tokens=8192,
                temperature=0.2,
                retries=2,
            )
            if not text:
                if attempt < max_batch_retries - 1:
                    logger.warning(
                        "video: %s: LLM returned empty (attempt %d/%d), retrying...",
                        batch_label or "batch",
                        attempt + 1,
                        max_batch_retries,
                    )
                    time.sleep(1 + attempt)
                    continue
                logger.warning(
                    "video: %s: LLM returned empty after %d attempts",
                    batch_label or "batch",
                    max_batch_retries,
                )
                break

            # Clear previous summaries before parsing new attempt
            if attempt > 0:
                for rec in batch:
                    if "summary" in rec.struct_data:
                        del rec.struct_data["summary"]

            # Parse TSV: index<TAB>summary
            parsed_count = 0
            for ln in text.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = parse_tsv_line(ln)
                if len(parts) < 2:
                    continue
                try:
                    idx = int(parts[0].strip())
                except ValueError:
                    continue
                if 0 <= idx < len(batch):
                    summary = html.unescape(parts[1].strip())
                    if summary:
                        batch[idx].struct_data["summary"] = summary
                        parsed_count += 1

            # Check success rate
            success_rate = parsed_count / len(batch) if batch else 0
            if success_rate >= success_rate_threshold:
                # Success - batch complete
                if attempt > 0:
                    logger.info(
                        "video: %s: retry succeeded - %d/%d summaries (%.1f%%)",
                        batch_label or "batch",
                        parsed_count,
                        len(batch),
                        success_rate * 100,
                    )
                break
            elif attempt < max_batch_retries - 1:
                # Partial failure - retry
                logger.warning(
                    "video: %s: only %d/%d summaries parsed (%.1f%%, < %.1f%% threshold, attempt %d/%d, retrying...)",
                    batch_label or "batch",
                    parsed_count,
                    len(batch),
                    success_rate * 100,
                    success_rate_threshold * 100,
                    attempt + 1,
                    max_batch_retries,
                )
                time.sleep(1 + attempt)
            else:
                # Final attempt failed
                logger.warning(
                    "video: %s: only %d/%d summaries parsed (%.1f%%) after %d attempts",
                    batch_label or "batch",
                    parsed_count,
                    len(batch),
                    success_rate * 100,
                    max_batch_retries,
                )

        # Return records that still don't have summaries
        failed = [rec for rec in batch if not rec.struct_data.get("summary")]
        return failed

    # Initial pass: process all records in normal batches
    failed_records: list[ShadowRecord] = []

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start : batch_start + batch_size]
        batch_label = f"batch at {batch_start}"
        failed = _process_batch(batch, batch_label)
        failed_records.extend(failed)

    # If we have failed records, retry them in smaller batches
    if failed_records:
        retry_batch_size = max(5, batch_size // 2)  # Smaller batches for retry (min 5)
        logger.info(
            "video: retrying %d failed records in smaller batches (size=%d)...",
            len(failed_records),
            retry_batch_size,
        )

        retry_failed: list[ShadowRecord] = []
        for retry_batch_start in range(0, len(failed_records), retry_batch_size):
            retry_batch = failed_records[retry_batch_start : retry_batch_start + retry_batch_size]
            retry_label = f"retry batch {retry_batch_start // retry_batch_size + 1}"
            failed = _process_batch(retry_batch, retry_label)
            retry_failed.extend(failed)

        if retry_failed:
            logger.warning(
                "video: %d records still missing summaries after retry with smaller batches",
                len(retry_failed),
            )

    summaries_count = sum(1 for r in records if r.struct_data.get("summary"))
    if summaries_count < len(records):
        missing = len(records) - summaries_count
        logger.warning(
            "video: generated %d/%d summaries (%d missing)", summaries_count, len(records), missing
        )
    else:
        logger.info("video: generated %d/%d summaries", summaries_count, len(records))


def _validate_video_segments(
    records: list[ShadowRecord],
    *,
    core_cfg: Config,
    batch_size: int = 50,
) -> list[ShadowRecord]:
    """Validate video segments using LLM to filter out non-valuable content."""
    if not records:
        return records

    from ..core.batch import batch_validate, filter_by_indices

    def build_prompt(batch: list[tuple[int, ShadowRecord]]) -> str:
        items = "\n".join(
            f"SEGMENT {idx}:\nVideo: {rec.struct_data.get('video_name', 'Video')}\n"
            f"Content: {rec.struct_data.get('quote', '')[:300]}"
            for idx, rec in batch
        )
        return f"""Evaluate which video segments are WORTH INDEXING for search.

SEGMENTS:
{items}

TASK: For each segment, decide if it provides educational/instructional value.

REJECT (NOT_VALUABLE) if the segment is:
- Technical/meta: "Can you hear me?", "Let me share my screen"
- Pure intro/outro: Just "Welcome everyone!" or "Thanks for watching"
- Housekeeping: "Take a break", "Submit questions in chat"
- Too short/fragmented: Incomplete thoughts, just "So...", "Now..."
- Promotional without teaching: Pure sales, program logistics

ACCEPT (VALUABLE) if the segment:
- Teaches a concept, principle, or method
- Shares an insight, story with lesson, or practical advice
- Contains searchable educational content

Return as TSV, one line per segment:
index<TAB>VALUABLE|NOT_VALUABLE

Rules:
- Use REAL tab characters (\\t), not "<TAB>"
- Maintain segment order
- Default to VALUABLE if uncertain (err on side of keeping)
"""

    valid_indices = batch_validate(
        records,
        core_cfg=core_cfg,
        prompt_builder=build_prompt,
        batch_size=batch_size,
        on_error="keep",
        label="video segments",
    )

    return filter_by_indices(records, valid_indices)


@register_plugin("video")
class VideoPlugin(IngestionPlugin):
    """Plugin for processing video transcripts."""

    @property
    def source_type(self) -> str:
        return "video"

    def load(self, assets_path: str) -> list[RawData]:
        """Load video transcripts from JSON files or youtube_transcript_api format."""
        source_dir = Path(assets_path)
        if not source_dir.exists():
            logger.warning("Video source directory does not exist: %s", assets_path)
            return []

        raw_data: list[RawData] = []

        # Look for JSON files (transcript format)
        for json_file in source_dir.glob("*.json"):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))

                # Extract YouTube ID and clean title from filename
                extracted_id, clean_filename = extract_youtube_id_from_filename(json_file.stem)

                # Handle different JSON formats:
                # 1. List format: [{"word": "...", "start": ..., "end": ...}, ...] (direct words list)
                # 2. Dict format: {"video_id": "...", "video_title": "...", "words": [...]}
                if isinstance(data, list):
                    # Direct list of words - use filename for metadata
                    words = data
                    video_id = extracted_id or json_file.stem
                    video_title = clean_filename or video_id
                elif isinstance(data, dict):
                    # Dictionary format - extract metadata and words
                    video_id = data.get("video_id") or extracted_id or json_file.stem
                    video_title = data.get("video_title") or clean_filename or video_id

                    # Handle different word formats in dict
                    words = data.get("words", [])
                    if not words:
                        # Try youtube_transcript_api format
                        words = data.get("transcript", [])
                else:
                    logger.warning(
                        "Unexpected JSON format in %s (expected list or dict)", json_file
                    )
                    continue

                if not words:
                    logger.warning("No words/transcript found in %s", json_file)
                    continue

                # Smart gluing into sentences
                sentences = smart_glue_words(words)

                # Create RawData for each sentence (or combine into one)
                content = " ".join(s["text"] for s in sentences)
                metadata: IngestionMetadata = {
                    "video_id": video_id,
                    "video_title": video_title,
                    "video_url": f"https://youtu.be/{video_id}",
                }

                raw_data.append(
                    RawData(
                        content=content,
                        source_type="video",
                        metadata=metadata,
                    )
                )

            except Exception as e:
                logger.error("Failed to load video file %s: %s", json_file, e)
                continue

        return raw_data

    def transform(
        self,
        data: list[RawData],
        enrichment_func: Callable[[str], GraphEnrichmentResult],
        taxonomy_path: Path | None = None,
        config: RagIngestionConfig | None = None,
        core_cfg: Config | None = None,
        **kwargs: object,
    ) -> list[ShadowRecord]:
        """Transform video data using sliding window chunking."""
        _ = taxonomy_path, kwargs
        if core_cfg is None:
            raise ValueError(
                "VideoPlugin.transform requires core_cfg (contextrouter.core.config.Config)"
            )
        shadow_records: list[ShadowRecord] = []

        logger.info("video: starting transform for %d video(s)", len(data))

        # Read LLM settings
        if config is None:
            from ..config import load_config

            config = load_config()
        llm_summary_enabled = config.video.llm_summary_enabled
        llm_summary_batch_size = config.video.llm_summary_batch_size
        llm_summary_max_sentences = config.video.llm_summary_max_sentences
        # Segment validation (filters non-valuable content)
        llm_segment_validation_enabled = config.video.llm_segment_validation_enabled
        llm_segment_validation_batch_size = config.video.llm_segment_validation_batch_size

        # Get persona name for summary generation
        persona_name = config.persona.persona_name.strip()
        if persona_name.lower() in {"", "speaker name"}:
            persona_name = ""

        for vid_idx, raw in enumerate(data, 1):
            video_id = raw.metadata.get("video_id", "unknown")
            video_title = raw.metadata.get("video_title", "Video")
            video_url = raw.metadata.get("video_url", f"https://youtu.be/{video_id}")

            # Prefer timing-aware sentences produced at preprocess stage
            sent_meta = raw.metadata.get("sentences")
            sentences_timed: list[dict[str, Any]] = []
            if isinstance(sent_meta, list):
                for s in sent_meta:
                    if not isinstance(s, dict):
                        continue
                    text = s.get("text")
                    start = s.get("start")
                    end = s.get("end")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                        continue
                    sentences_timed.append(
                        {"text": text.strip(), "start": float(start), "end": float(end)}
                    )

            if sentences_timed:
                total_windows = max(0, len(sentences_timed) - 2)
                logger.info(
                    "video: processing %s [%d/%d] - %d timed sentences -> ~%d windows",
                    video_title,
                    vid_idx,
                    len(data),
                    len(sentences_timed),
                    total_windows,
                )

                # Sliding window: stride=1, size=3 sentences
                for i in range(total_windows):
                    if i > 0 and i % 100 == 0:
                        logger.info(
                            "video: %s - processed %d/%d windows...", video_title, i, total_windows
                        )

                    window = sentences_timed[i : i + 3]
                    window_text = " ".join(w.get("text", "") for w in window).strip()
                    if not window_text:
                        continue

                    start_seconds = int(max(0, window[0].get("start", 0.0)))
                    ts = format_timestamp(start_seconds)

                    # Enrichment
                    keywords, summary, parent_categories = get_graph_enrichment(
                        text=window_text, enrichment_func=enrichment_func
                    )
                    initial_keywords = raw.metadata.get("keywords", [])
                    if not isinstance(initial_keywords, list):
                        initial_keywords = []
                    keywords = list(dict.fromkeys([*initial_keywords, *keywords]))[:10]

                    # Build input_text with QA-style explicit enrichment format
                    input_text = self._build_input_text(
                        content=window_text,
                        keywords=keywords,
                        summary=summary,
                        parent_categories=parent_categories,
                    )

                    clean_quote = normalize_clean_text(window_text)
                    # Unicode normalization happens during preprocess stage
                    record_id = generate_id(video_id, str(start_seconds), window_text[:50])

                    struct_data: VideoStructData = {
                        "source_type": "video",
                        "video_name": video_title,
                        "video_id": video_id,
                        "video_url": f"{video_url}?t={start_seconds}",
                        "timestamp": ts,
                        "timestamp_seconds": start_seconds,
                        "keywords": clean_str_list(keywords, limit=10),
                        "quote": clean_quote,
                    }

                    shadow_records.append(
                        ShadowRecord(
                            id=record_id,
                            input_text=input_text,
                            struct_data=struct_data,
                            title=f"{video_title} [{ts}]",
                            source_type="video",
                        )
                    )

                logger.info(
                    "video: %s - created %d shadow records",
                    video_title,
                    len([r for r in shadow_records if r.struct_data.get("video_id") == video_id]),
                )
                continue

            # Fallback: split content into sentences (no timing data)
            sentences = [s.strip() for s in raw.content.split(". ") if s.strip()]
            total_windows = len(sentences) - 2

            logger.info(
                "video: processing %s [%d/%d] - %d sentences -> ~%d windows (no timing data)",
                video_title,
                vid_idx,
                len(data),
                len(sentences),
                total_windows,
            )

            # Sliding window: stride=1, size=3 sentences
            for i in range(total_windows):
                if i > 0 and i % 100 == 0:
                    logger.info(
                        "video: %s - processed %d/%d windows...", video_title, i, total_windows
                    )

                window = sentences[i : i + 3]
                window_text = ". ".join(window)
                if not window_text.endswith("."):
                    window_text += "."

                # Estimate start time (rough approximation)
                start_seconds = int(i * 5)  # ~5 seconds per sentence estimate
                ts = format_timestamp(start_seconds)

                # Enrichment
                keywords, summary, parent_categories = get_graph_enrichment(
                    text=window_text, enrichment_func=enrichment_func
                )
                initial_keywords = raw.metadata.get("keywords", [])
                if not isinstance(initial_keywords, list):
                    initial_keywords = []
                keywords = list(dict.fromkeys([*initial_keywords, *keywords]))[:10]

                # Build input_text with QA-style explicit enrichment format
                input_text = self._build_input_text(
                    content=window_text,
                    keywords=keywords,
                    summary=summary,
                    parent_categories=parent_categories,
                )

                # Clean quote text: replace newlines with spaces for better UI display
                # Unicode normalization happens during preprocess stage
                clean_quote = normalize_clean_text(window_text)

                # Generate ID
                record_id = generate_id(video_id, str(start_seconds), window_text[:50])

                # Build struct_data matching frontend schema
                struct_data: VideoStructData = {
                    "source_type": "video",
                    # snake_case for retriever/citations compatibility
                    "video_name": video_title,
                    "video_id": video_id,
                    "video_url": f"{video_url}?t={start_seconds}",
                    "timestamp": ts,
                    "timestamp_seconds": start_seconds,
                    "keywords": clean_str_list(keywords, limit=10),
                    "quote": clean_quote,
                }

                shadow_records.append(
                    ShadowRecord(
                        id=record_id,
                        input_text=input_text,
                        struct_data=struct_data,
                        title=f"{video_title} [{ts}]",
                        source_type="video",
                    )
                )

        logger.info("video: transform complete - %d total shadow records", len(shadow_records))

        # Optional: Validate segments with LLM (expensive, off by default)
        if llm_segment_validation_enabled and shadow_records:
            shadow_records = _validate_video_segments(
                shadow_records,
                core_cfg=core_cfg,
                batch_size=llm_segment_validation_batch_size,
            )

        # Generate LLM summaries for all records (batch processing)
        if llm_summary_enabled and shadow_records:
            logger.info(
                "video: generating LLM summaries for %d records (persona=%s)...",
                len(shadow_records),
                persona_name or "generic",
            )
            _generate_video_summaries(
                shadow_records,
                core_cfg=core_cfg,
                batch_size=llm_summary_batch_size,
                max_sentences=llm_summary_max_sentences,
                persona_name=persona_name,
            )

        return shadow_records

    def _build_input_text(
        self,
        content: str,
        keywords: list[str],
        summary: str | None = None,
        parent_categories: list[str] | None = None,
    ) -> str:
        """Build shadow context input_text with QA-style explicit enrichment format.

        Args:
            content: Video window content
            keywords: Graph enrichment keywords
            summary: Graph enrichment summary (relation descriptions)
            parent_categories: Taxonomy categories from graph enrichment

        Returns:
            Formatted input_text string with explicit Categories: and Additional Knowledge: headers
        """
        parts = [content]

        # Add taxonomy categories from graph enrichment (QA-style)
        if parent_categories:
            cats = [c for c in parent_categories if isinstance(c, str) and c.strip()]
            if cats:
                cat_str = ", ".join(cats[:5])
                parts.append(f"Categories: {cat_str}")

        # Add natural language enrichment for keywords (QA-style)
        if keywords:
            top_keywords = keywords[:10]
            if len(top_keywords) == 1:
                parts.append(f"Additional Knowledge: This text is related to {top_keywords[0]}.")
            elif len(top_keywords) <= 3:
                keywords_str = ", ".join(top_keywords[:-1])
                parts.append(
                    f"Additional Knowledge: This text is related to {keywords_str} and {top_keywords[-1]}."
                )
            else:
                keywords_str = ", ".join(top_keywords[:5])
                parts.append(
                    f"Additional Knowledge: This text is related to {keywords_str}, and other concepts."
                )

        # Add summary if available (graph relations)
        if isinstance(summary, str) and summary.strip():
            parts.append(f"Additional Knowledge: {summary.strip()}")

        return "\n".join(parts)
