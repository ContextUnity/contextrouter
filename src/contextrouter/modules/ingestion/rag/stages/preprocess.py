"""Stage 1: Raw -> CleanText (persisted per source type)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from contextrouter.core import Config

from ..config import get_assets_paths, get_plugin_source_dir
from ..core import IngestionMetadata, RawData
from ..core.utils import (
    normalize_ambiguous_unicode,
    parallel_map,
    resolve_workers,
)
from ..plugins import QAPlugin
from ..plugins.book import BookPlugin
from ..plugins.text import TextPlugin
from ..plugins.video import (
    extract_youtube_id_from_filename,
    smart_glue_words,
)
from ..plugins.web import WebPlugin
from ..settings import RagIngestionConfig
from ..utils.llm import llm_generate
from .store import write_raw_data_jsonl

logger = logging.getLogger(__name__)


def _resolve_preprocess_model(core_cfg: Config) -> str:
    return core_cfg.models.ingestion.preprocess.model.strip()


def preprocess_to_clean_text(
    *,
    core_cfg: Config,
    config: RagIngestionConfig,
    only_types: list[str],
    overwrite: bool = True,
    workers: int = 1,
) -> dict[str, Path]:
    """Preprocess raw sources into clean_text JSONL per type.

    Returns a mapping {source_type: output_path}.
    """
    paths = get_assets_paths(config)
    preprocess_model = _resolve_preprocess_model(core_cfg)
    logger.info("preprocess: model=%s", preprocess_model)
    outputs: dict[str, Path] = {}

    def _run_one(t: str) -> tuple[str, Path, int]:
        import time

        start = time.perf_counter()
        logger.info("preprocess: processing type=%s", t)
        model_name = preprocess_model.split("/")[-1]
        out_path = paths["clean_text"] / f"{t}_{model_name}.jsonl"
        items: list[RawData]

        if t == "video":
            items = _preprocess_video(
                source_dir=get_plugin_source_dir("video", config),
                config=config,
                core_cfg=core_cfg,
            )
        elif t == "book":
            items = _preprocess_book(
                source_dir=get_plugin_source_dir("book", config),
                config=config,
                core_cfg=core_cfg,
            )
        elif t == "qa":
            items = _preprocess_qa(
                source_dir=get_plugin_source_dir("qa", config),
                config=config,
                core_cfg=core_cfg,
            )
        elif t == "web":
            items = WebPlugin().load(str(get_plugin_source_dir("web", config)), config=config)
            _populate_web_llm_summaries(items, config=config, core_cfg=core_cfg)
        elif t == "knowledge":
            items = TextPlugin().load(str(get_plugin_source_dir("knowledge", config)))
        else:
            logger.warning("preprocess: unknown type=%s (skipped)", t)
            return (t, out_path, 0)

        count = write_raw_data_jsonl(items, out_path, overwrite=overwrite)
        elapsed = time.perf_counter() - start
        logger.info(
            "preprocess: wrote=%d type=%s path=%s (%.1fs)",
            count,
            t,
            out_path.name,
            elapsed,
        )
        return (t, out_path, count)

    w = resolve_workers(config=config, workers=workers)
    if w > 1 and len(only_types) > 1:
        logger.info("preprocess: parallelism workers=%d", w)

    results = parallel_map(only_types, _run_one, workers=w, ordered=False, swallow_exceptions=False)
    for r in results:
        if not r:
            continue
        tt, out_path, _ = r
        outputs[tt] = out_path

    return outputs


def _preprocess_book(
    *, source_dir: Path, config: RagIngestionConfig, core_cfg: Config
) -> list[RawData]:
    """Book preprocessing using BookPlugin for PDF extraction and chapter detection."""
    plugin = BookPlugin()
    items = plugin.load(str(source_dir))
    return items


def _preprocess_qa(
    *, source_dir: Path, config: RagIngestionConfig, core_cfg: Config
) -> list[RawData]:
    """QA preprocessing (optional LLM speaker detection + interruption merge + question filtering).

    If enabled:
    - Detects speakers and splits into interactions
    - Merges short interruptions
    - Filters out meaningless interactions (greetings, pleasantries)
    - Stores structured turns into metadata["interactions"]
    - Rewrites RawData.content to a cleaned transcript (no speaker tags) for taxonomy/graph.
    """
    speaker_enabled = bool(config.qa.llm_speaker_detect_enabled)
    question_filter_enabled = bool(config.qa.llm_question_filter_enabled)

    plugin = QAPlugin()
    plugin.set_core_cfg(core_cfg)
    items = plugin.load(str(source_dir))
    if not speaker_enabled:
        return items

    logger.info("preprocess(qa): speaker detection enabled")
    if question_filter_enabled:
        logger.info("preprocess(qa): question filtering enabled")

    for i, raw in enumerate(items, start=1):
        title = ""
        if isinstance(raw.metadata, dict):
            title = str(
                raw.metadata.get("session_title") or raw.metadata.get("source_title") or ""
            )[:80]
        logger.info("preprocess(qa): session %d/%d title=%s", i, len(items), title or "Untitled")

        interactions = plugin.split_by_speakers_llm(raw.content)
        if interactions is None:
            raise RuntimeError("QA speaker detection returned None; expected list of interactions")
        if not interactions:
            interactions = [{"speaker": "Unknown", "text": raw.content}]

        interactions = plugin.merge_interruptions(interactions)

        # Filter out meaningless interactions (greetings, pleasantries, low-value patterns)
        if question_filter_enabled:
            interactions = _filter_meaningless_interactions(interactions, core_cfg=core_cfg)

        # Persist structured turns for shadow stage reuse
        if not isinstance(raw.metadata, dict):
            raw.metadata = {}
        raw.metadata["interactions"] = interactions

        # Identify session host/interviewer (dominant NON-persona speaker)
        # Used for fallback headers: "Speaker Name sharing thoughts on 'topic' with {Host}"
        session_host = _identify_session_host(interactions, config, core_cfg=core_cfg)
        if session_host:
            raw.metadata["session_host"] = session_host

        # Rewrite content for taxonomy/graph: plain text only, no speaker tags
        content_text = " ".join(
            str(x.get("text") or "").strip()
            for x in interactions
            if str(x.get("text") or "").strip()
        )

        # Normalize ambiguous unicode during preprocess
        raw.content = normalize_ambiguous_unicode(content_text)

    return items


def _identify_session_host(
    interactions: list[dict[str, str]], config: RagIngestionConfig, *, core_cfg: Config
) -> str | None:
    """Identify the host/interviewer for the session using LLM analysis.

    The host is the person who introduces topics, asks questions, and guides
    the conversation - NOT a guest speaker who gives long talks.

    Uses LLM to analyze speaker patterns and roles, with interaction-count
    heuristic as fallback.

    Args:
        interactions: List of {speaker, text} dicts for the session
        config: Ingestion config with persona and corrections

    Returns:
        Host/interviewer name (corrected), or None if can't determine
    """
    from collections import Counter

    if not interactions:
        return None

    # Load corrections (from video.corrections or qa.corrections)
    corrections: dict[str, str] = {}
    corrections.update(config.video.corrections)
    corrections.update(config.qa.corrections)

    def apply_corrections(name: str) -> str:
        for wrong, correct in corrections.items():
            if wrong.casefold() in name.casefold():
                name = name.replace(wrong, correct)
        return name.strip()

    # Build speaker stats: word counts and interaction counts
    speaker_word_counts: Counter[str] = Counter()
    speaker_interaction_counts: Counter[str] = Counter()
    norm_to_corrected: dict[str, str] = {}

    for it in interactions:
        if not isinstance(it, dict):
            continue
        sp = it.get("speaker")
        tx = it.get("text")
        if not sp or not isinstance(sp, str) or not tx or not isinstance(tx, str):
            continue
        corrected = apply_corrections(sp.strip()) or "Unknown"
        norm = corrected.casefold()
        norm_to_corrected.setdefault(norm, corrected)
        speaker_word_counts[norm] += len(tx.split())
        speaker_interaction_counts[norm] += 1

    if not speaker_word_counts:
        return None

    # Get persona name to exclude from host search
    persona_name = config.persona.persona_name.strip()
    pn_norm = (
        persona_name.casefold()
        if persona_name and persona_name.lower() not in {"", "speaker name"}
        else ""
    )

    # Filter to non-persona speakers
    non_persona_speakers = [
        norm for norm in speaker_word_counts.keys() if norm != pn_norm and norm != "unknown"
    ]

    if not non_persona_speakers:
        return None

    # If only one non-persona speaker, that's the host
    if len(non_persona_speakers) == 1:
        return norm_to_corrected.get(non_persona_speakers[0])

    # Multiple non-persona speakers: use LLM to identify host
    llm_host_detect = config.qa.llm_host_detect_enabled

    if llm_host_detect:
        host = _identify_host_llm(
            interactions,
            non_persona_speakers,
            norm_to_corrected,
            pn_norm,
            core_cfg=core_cfg,
        )
        if host:
            return host

    # Fallback: use interaction-count heuristic
    # Host has many short interactions (intros, questions, transitions)
    # Guest speakers have few long interactions (talks, explanations)
    # Score = interaction_count / word_count (higher = more host-like)
    host_scores = []
    for norm in non_persona_speakers:
        words = speaker_word_counts[norm]
        interactions_count = speaker_interaction_counts[norm]
        # Avoid division by zero
        if words > 0:
            score = interactions_count / words * 1000  # Scale for readability
            host_scores.append((norm, score))

    if not host_scores:
        return None

    # Return speaker with highest host score (many short interactions)
    best_host_norm = max(host_scores, key=lambda x: x[1])[0]
    return norm_to_corrected.get(best_host_norm)


def _identify_host_llm(
    interactions: list[dict[str, str]],
    candidates: list[str],
    norm_to_corrected: dict[str, str],
    persona_norm: str,
    *,
    core_cfg: Config,
) -> str | None:
    """Use LLM to identify the host from multiple speaker candidates.

    Analyzes first portion of transcript to understand speaker roles.
    """

    # Build sample of first ~2000 chars showing speaker patterns
    sample_lines = []
    char_count = 0
    for it in interactions:
        if char_count > 2000:
            break
        sp = it.get("speaker", "Unknown")
        tx = it.get("text", "")[:200]  # Truncate long texts
        line = f"{sp}: {tx}"
        sample_lines.append(line)
        char_count += len(line)

    if not sample_lines:
        return None

    transcript_sample = "\n".join(sample_lines)
    candidate_names = [norm_to_corrected.get(c, c) for c in candidates]

    prompt = f"""Analyze this transcript and identify who is the HOST/MODERATOR.

The host is the person who:
- Introduces topics and transitions between segments
- Asks questions to guide the discussion
- Has many SHORT interactions (intros, questions, acknowledgments)

The host is NOT:
- A guest speaker who gives long talks/explanations
- The main expert/teacher (that's the persona)

TRANSCRIPT SAMPLE:
{transcript_sample}

CANDIDATE SPEAKERS (excluding the main expert):
{", ".join(candidate_names)}

Which of these candidates is most likely the HOST/MODERATOR?
Return ONLY the exact name from the candidates list, nothing else.

HOST:"""

    try:
        preprocess_model = _resolve_preprocess_model(core_cfg)
        result = llm_generate(
            core_cfg=core_cfg,
            prompt=prompt,
            model=preprocess_model,
            max_tokens=50,
            temperature=0.0,
        )
        if isinstance(result, str):
            result = result.strip()
            # Match result to candidates
            for cand in candidates:
                corrected_name = norm_to_corrected.get(cand, cand)
                if result.casefold() == corrected_name.casefold() or result.casefold() == cand:
                    return corrected_name
    except Exception as e:
        logger.debug("LLM host detection failed: %s (using heuristic)", e)

    return None


def _filter_meaningless_interactions(
    interactions: list[dict[str, str]], *, core_cfg: Config
) -> list[dict[str, str]]:
    """Filter out meaningless interactions (greetings, pleasantries, low-value conversational patterns).

    Uses smart batching with overlap to preserve conversational context:
    - Processes interactions with overlapping windows to avoid cutting mid-conversation
    - Overlap ensures interactions at batch boundaries are evaluated in proper context
    - Resolves conflicts by preferring decisions from batches where interaction is in the middle

    Uses LLM to determine if each interaction contains meaningful content.
    Removes interactions that are just greetings, pleasantries, or filler.

    Args:
        interactions: List of {speaker, text} dicts

    Returns:
        Filtered list with only meaningful interactions
    """
    if not interactions:
        return interactions

    # Smart batching with overlap to preserve conversational context
    batch_size = 20
    overlap_size = max(5, batch_size // 4)  # 5 interactions or ~25% overlap, whichever is larger
    step_size = batch_size - overlap_size  # Step forward by effective batch size

    all_decisions: dict[
        int, list[tuple[int, str]]
    ] = {}  # interaction_idx -> [(batch_offset, decision), ...]
    batch_num = 0

    # Process with overlapping windows
    for batch_start in range(0, len(interactions), step_size):
        batch_end = min(batch_start + batch_size, len(interactions))
        batch = interactions[batch_start:batch_end]

        if not batch:
            break

        batch_num += 1

        batch_data = []
        for i, it in enumerate(batch):
            text = str(it.get("text", "")).strip()
            if not text:
                continue
            # Store global index for tracking
            batch_data.append({"idx": batch_start + i, "text": text})

        if not batch_data:
            continue

        # Build prompt to filter batch
        items_text = "\n".join([f"CHUNK {item['idx']}: {item['text']}" for item in batch_data])

        prompt = f"""Filter conversational interactions to keep only meaningful content.

For each chunk, decide:
- KEEP: Contains a question, inquiry, substantive statement, or meaningful content
- DROP: Only greetings, pleasantries, filler, or low-value conversational patterns

Examples of DROP:
- "Hey guys, welcome to the program!"
- "Thank you" (standalone)
- "Great" (standalone)
- "Yeah" (standalone)
- "Um", "So" (as filler), "Well" (just filler)
- Only small talk without substance
- Generic question labels: "Question from [Name]" without actual question

Examples of KEEP:
- "What are the three principles we'll be covering?"
- "Can you explain how persistence relates to achieving goals?"
- Substantive statements that set up answers
- Questions or inquiries

Return as TSV, one line per chunk:
index\tKEEP|DROP

Rules:
- Use REAL tab characters (\t), not "<TAB>"
- Maintain order (0, 1, 2, ...)

CHUNKS:
{items_text}
"""

        try:
            preprocess_model = _resolve_preprocess_model(core_cfg)
            text = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=preprocess_model,
                max_tokens=1024,
                temperature=0.1,
                parse_json=False,
            )
        except Exception as e:
            logger.warning(
                "QA question filtering failed for batch at %d: %s (keeping all)",
                batch_start,
                e,
            )
            # On failure, keep all interactions (safer) - add KEEP decisions
            for i, _ in enumerate(batch):
                global_idx = batch_start + i
                all_decisions.setdefault(global_idx, []).append((i, "KEEP"))
            continue

        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "QA question filtering returned empty for batch at %d (keeping all)",
                batch_start,
            )
            # Keep all interactions - add KEEP decisions
            for i, _ in enumerate(batch):
                global_idx = batch_start + i
                all_decisions.setdefault(global_idx, []).append((i, "KEEP"))
            continue

        # Parse decisions and store with global indices
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            try:
                global_idx = int(parts[0].strip())  # This is now the global interaction index
            except ValueError:
                continue
            decision = parts[1].strip().upper()
            if decision not in {"KEEP", "DROP"}:
                continue

            # Track which batch offset this decision came from (distance from batch start/end)
            batch_local_idx = global_idx - batch_start
            batch_offset_from_start = batch_local_idx
            batch_offset_from_end = len(batch) - 1 - batch_local_idx
            # Lower offset means closer to edge - prefer decisions from middle of batch
            batch_offset = min(batch_offset_from_start, batch_offset_from_end)

            if global_idx not in all_decisions:
                all_decisions[global_idx] = []
            all_decisions[global_idx].append((batch_offset, decision))

    # Resolve overlapping decisions: prefer decisions from batches where interaction is in the middle
    # If multiple batches processed the same interaction, use the decision from the batch where it's
    # farthest from the edge (has highest batch_offset value)
    final_decisions: dict[int, str] = {}
    for global_idx, decision_list in all_decisions.items():
        # Sort by batch_offset descending (prefer decisions from middle of batches)
        decision_list.sort(key=lambda x: x[0], reverse=True)
        # Use the decision from the batch where interaction was most central
        final_decisions[global_idx] = decision_list[0][1]

    # Build filtered list based on final decisions
    filtered: list[dict[str, str]] = []
    for i, it in enumerate(interactions):
        decision = final_decisions.get(i, "KEEP")  # Default to KEEP if no decision
        if decision == "KEEP":
            filtered.append(it)

    dropped = len(interactions) - len(filtered)
    if dropped > 0:
        logger.info("    Filtered %d/%d meaningless interactions", dropped, len(interactions))

    return filtered


def _populate_web_llm_summaries(
    items: list[RawData], *, config: RagIngestionConfig, core_cfg: Config
) -> None:
    """Ensure each web RawData has a short summary in metadata for citations/UI."""
    if not config.web.llm_summary_enabled:
        logger.info("preprocess(web): llm_summary_enabled=false (skipping)")
        return

    max_chars = config.web.llm_summary_max_chars
    out_chars = config.web.llm_summary_output_chars

    for item in items:
        if item.source_type != "web":
            continue
        if not isinstance(item.metadata, dict):
            continue
        existing = item.metadata.get("summary")
        if isinstance(existing, str) and existing.strip():
            continue
        url = item.metadata.get("url") if isinstance(item.metadata.get("url"), str) else ""
        title = item.metadata.get("title") if isinstance(item.metadata.get("title"), str) else ""
        content = item.content if isinstance(item.content, str) else ""
        if not content.strip():
            continue

        prompt = (
            "You are writing a short description for a web page to be shown in citations.\n"
            "Return ONE concise sentence (no bullet points, no quotes), <= 25 words.\n\n"
            f"URL: {url}\n"
            f"TITLE: {title}\n\n"
            f"CONTENT:\n{content[:max_chars]}\n"
        )
        try:
            preprocess_model = _resolve_preprocess_model(core_cfg)
            summary = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=preprocess_model,
                max_tokens=256,
                temperature=0.2,
                parse_json=False,
            )
        except Exception as e:
            logger.warning("preprocess(web): summary LLM failed (url=%s): %s", url, e)
            continue

        if isinstance(summary, str):
            s = " ".join(summary.strip().split())
            if out_chars > 0 and len(s) > out_chars:
                s = s[: out_chars - 1].rstrip() + "…"
            if s:
                item.metadata["summary"] = s


def _preprocess_video(
    *, source_dir: Path, config: RagIngestionConfig, core_cfg: Config
) -> list[RawData]:
    """Video preprocessing that preserves timing information.

    Produces one RawData per transcript file with:
    - content: full transcript text
    - metadata.sentences: list[{text,start,end}] (seconds)

    If [video].llm_clean_enabled is True, performs LLM-based cleaning to remove
    filler, garbled text, and promotional content while preserving timing.
    """
    if not source_dir.exists():
        logger.warning("Video source directory does not exist: %s", source_dir)
        return []

    llm_clean_enabled = config.video.llm_clean_enabled
    llm_batch_size = config.video.llm_clean_batch_size
    corrections = config.video.corrections

    out: list[RawData] = []

    json_files = sorted(source_dir.glob("*.json"))
    total_files = len(json_files)

    for file_idx, json_file in enumerate(json_files, 1):
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read video JSON %s: %s", json_file, e)
            continue

        extracted_id, clean_filename = extract_youtube_id_from_filename(json_file.stem)

        words: list[dict[str, Any]] = []
        video_id = extracted_id or json_file.stem
        video_title = clean_filename or video_id

        if isinstance(payload, list):
            words = payload
        elif isinstance(payload, dict):
            video_id = payload.get("video_id") or video_id
            video_title = payload.get("video_title") or video_title
            w = payload.get("words") or payload.get("transcript") or []
            if isinstance(w, list):
                words = w

        if not words:
            logger.warning("No words/transcript found in %s", json_file)
            continue

        logger.info("preprocess(video): %d/%d %s", file_idx, total_files, video_title)

        sentences = smart_glue_words(words)

        # Optional LLM cleaning
        if llm_clean_enabled:
            sentences = _llm_clean_video_sentences(
                sentences,
                corrections=corrections,
                batch_size=llm_batch_size,
                video_title=video_title,
                core_cfg=core_cfg,
            )
        elif corrections:
            # Deterministic corrections should not depend on LLM; apply to all sentences.
            for s in sentences:
                if isinstance(s, dict):
                    s["text"] = _apply_deterministic_corrections(
                        str(s.get("text") or ""), corrections
                    )

        sentence_texts = [
            str(s.get("text") or "").strip() for s in sentences if str(s.get("text") or "").strip()
        ]
        transcript = " ".join(sentence_texts).strip()

        # Normalize ambiguous unicode during preprocess
        transcript = normalize_ambiguous_unicode(transcript)
        video_title = normalize_ambiguous_unicode(str(video_title))  # Normalize video title too

        md: IngestionMetadata = {
            "video_id": str(video_id),
            "video_title": video_title,
            "video_url": f"https://youtu.be/{video_id}",
            # CleanText enrichment for later stages
            "sentences": [
                {
                    "text": normalize_ambiguous_unicode(
                        str(s.get("text") or "")
                    ),  # Normalize sentence text too
                    "start": float(s.get("start") or 0.0),
                    "end": float(s.get("end") or 0.0),
                }
                for s in sentences
                if isinstance(s, dict)
            ],
        }

        out.append(RawData(content=transcript, source_type="video", metadata=md))

    return out


def _llm_clean_video_sentences(
    sentences: list[dict[str, Any]],
    *,
    corrections: dict[str, str],
    batch_size: int,
    video_title: str,
    core_cfg: Config,
) -> list[dict[str, Any]]:
    """LLM-based cleaning of video transcript sentences.

    Uses smart batching with overlap to preserve thought boundaries:
    - Processes sentences with overlapping windows to avoid cutting mid-thought
    - Overlap ensures sentences at batch boundaries are evaluated in context
    - Resolves conflicts by preferring decisions from batches where sentence is in the middle

    For each sentence, the LLM decides:
    - KEEP: meaningful content (keep the sentence)
    - DROP: filler/noise/promo/unintelligible/repetition (remove the sentence)

    Deterministic corrections are applied locally (never gated by the LLM).
    Preserves timing (start/end) for kept sentences.
    """
    if not sentences:
        return []

    # Build corrections prompt section
    corrections_section = ""
    if corrections:
        lines = ["Known transcription corrections to apply (deterministic string replacements):"]
        for orig, repl in corrections.items():
            lines.append(f'- "{orig}" → "{repl}"')
        corrections_section = "\n".join(lines)

    # Smart batching with overlap to preserve thought boundaries
    # Overlap ensures sentences at batch edges are evaluated in proper context
    overlap_size = max(10, batch_size // 6)  # 10 sentences or ~16% overlap, whichever is larger
    step_size = batch_size - overlap_size  # Step forward by effective batch size

    all_decisions: dict[
        int, list[tuple[int, str]]
    ] = {}  # sentence_idx -> [(batch_offset, decision), ...]
    batch_num = 0
    # Correct total: ceil division for overlapping windows
    total_batches = (len(sentences) + step_size - 1) // step_size

    # Process with overlapping windows
    for batch_start in range(0, len(sentences), step_size):
        batch_end = min(batch_start + batch_size, len(sentences))
        batch = sentences[batch_start:batch_end]

        if not batch:
            break

        batch_num += 1

        # Create batch with global indices (for tracking)
        batch_texts = [
            {"idx": batch_start + i, "text": s.get("text", "")} for i, s in enumerate(batch)
        ]

        logger.info(
            "preprocess(video): llm clean batch %d/%d (%d sentences overlap=%d)",
            batch_num,
            total_batches,
            len(batch),
            overlap_size,
        )

        prompt = f"""Clean a video transcript.

You will receive a JSON array of sentences: [{{"idx": <int>, "text": <string>}}, ...]

For each sentence, decide:
- KEEP: meaningful content (keep the sentence)
- DROP: filler/noise/promo/unintelligible/repetition/greetings/closings (remove the sentence)

Examples of DROP:
- Filler: "Um", "Uh", "You know", "Like", "So" (when standalone or leading to nothing), "Well" (as standalone sentences)
- Noise: Garbled text, unintelligible words, transcription errors
- Promo: "Don't forget to subscribe", "Click the bell icon", "Check out my website"
- Repetition: Exact duplicates or near-duplicates
- Greetings: "Welcome everybody", "Welcome to...", "Thanks for watching", "Hey everyone"
- Closings: "See you next time", "Thanks for watching", "Until next time"
- Low-value: "Okay", "Yeah", "Right" (as standalone acknowledgments)
- Small talk: Conversational filler without substance
- Standalone "So" at start: "So..." followed by incomplete thought or just "So," with pause

Examples of KEEP:
- Substantive content: "The first principle is to have a clear goal."
- Questions: "What are the three steps to success?"
- Explanations: "This method works because it aligns with your values."
- Instructions: "Here's how to implement this strategy."
- Stories/examples: "I once met someone who..."
- Concepts/principles: "The key is understanding your why."

{corrections_section}

OUTPUT FORMAT (STRICT, no exceptions):
- Return exactly {len(batch_texts)} lines
- One line per input sentence
- Each line: <idx>\t<KEEP|DROP>
- idx must match input idx
- No extra text, no markdown, no explanations

INPUT JSON:
{json.dumps(batch_texts, ensure_ascii=False)}
"""

        try:
            preprocess_model = _resolve_preprocess_model(core_cfg)
            raw = llm_generate(
                core_cfg=core_cfg,
                prompt=prompt,
                model=preprocess_model,
                max_tokens=2048,
                temperature=0.1,
                parse_json=False,
            )
        except Exception as e:
            logger.warning(
                "LLM clean failed for batch at %d: %s (keeping original)",
                batch_start,
                e,
            )
            # On failure, keep all sentences - add KEEP decisions
            for i, _ in enumerate(batch):
                global_idx = batch_start + i
                all_decisions.setdefault(global_idx, []).append((i, "KEEP"))
            continue

        if not isinstance(raw, str) or not raw.strip():
            logger.warning(
                "LLM clean returned empty output for batch at %d (keeping original)",
                batch_start,
            )
            # Keep all sentences - add KEEP decisions
            for i, _ in enumerate(batch):
                global_idx = batch_start + i
                all_decisions.setdefault(global_idx, []).append((i, "KEEP"))
            continue

        # Parse tab-separated lines.
        # Store decisions with global sentence indices
        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        for ln in lines:
            parts = ln.split("\t")
            if len(parts) not in {2, 3}:
                continue
            idx_s, action_s = parts[0], parts[1]
            try:
                global_idx = int(idx_s)  # This is now the global sentence index
            except ValueError:
                continue
            action = action_s.strip().upper()
            if action not in {"KEEP", "DROP"}:
                continue

            # Track which batch offset this decision came from (distance from batch start/end)
            batch_local_idx = global_idx - batch_start
            batch_offset_from_start = batch_local_idx
            batch_offset_from_end = len(batch) - 1 - batch_local_idx
            # Lower offset means closer to edge - prefer decisions from middle of batch
            batch_offset = min(batch_offset_from_start, batch_offset_from_end)

            if global_idx not in all_decisions:
                all_decisions[global_idx] = []
            all_decisions[global_idx].append((batch_offset, action))

    # Resolve overlapping decisions: prefer decisions from batches where sentence is in the middle
    # If multiple batches processed the same sentence, use the decision from the batch where it's
    # farthest from the edge (has highest batch_offset value)
    final_decisions: dict[int, str] = {}
    for global_idx, decision_list in all_decisions.items():
        # Sort by batch_offset descending (prefer decisions from middle of batches)
        decision_list.sort(key=lambda x: x[0], reverse=True)
        # Use the decision from the batch where interaction was most central
        final_decisions[global_idx] = decision_list[0][1]

    # Build cleaned list based on final decisions
    cleaned: list[dict[str, Any]] = []
    for i, orig_sentence in enumerate(sentences):
        action = final_decisions.get(i, "KEEP")  # Default to KEEP if no decision

        if action == "DROP":
            continue  # Skip this sentence

        # Always apply deterministic replacements; preserve timing.
        txt = str(orig_sentence.get("text") or "")
        if corrections:
            txt = _apply_deterministic_corrections(txt, corrections)
        cleaned.append(
            {
                "text": txt,
                "start": orig_sentence.get("start", 0.0),
                "end": orig_sentence.get("end", 0.0),
            }
        )

    dropped = len(sentences) - len(cleaned)
    logger.info(
        "preprocess(video): llm clean complete: %d -> %d sentences (dropped %d)",
        len(sentences),
        len(cleaned),
        dropped,
    )
    return cleaned


def _apply_deterministic_corrections(text: str, corrections: dict[str, str]) -> str:
    """Apply configured string replacements deterministically (exact substring matches)."""
    out = text
    for orig, repl in corrections.items():
        if isinstance(orig, str) and orig:
            out = out.replace(orig, str(repl))
    return out


__all__ = ["preprocess_to_clean_text"]
