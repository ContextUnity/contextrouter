"""Style processor for persona profile generation."""

from __future__ import annotations

import logging
import random
from pathlib import Path

from contextrouter.core import Config

from ..core.types import RawData
from ..utils.llm import llm_generate

logger = logging.getLogger(__name__)


def generate_persona_profile(
    all_data: list[RawData],
    output_path: Path,
    *,
    core_cfg: Config,
    persona_name: str = "Speaker Name",
    bio_text: str | None = None,
    sample_count: int = 50,
    max_chars_per_sample: int = 500,
    max_output_tokens: int = 8192,
) -> None:
    """Generate a persona system instruction from content samples.

    Args:
        all_data: List of all RawData objects
        output_path: Path to save the persona.txt file
        persona_name: Explicit persona name to use (prevents wrong-name inference)
        bio_text: Optional full bio text to include (high-signal grounding)
        sample_count: Number of random samples to use for tone
        max_chars_per_sample: Max chars per sample
    """
    # Filter to spoken content
    filtered = [d for d in all_data if d.source_type in ["video", "qa", "knowledge"]]

    if not filtered:
        logger.warning("No video/qa/knowledge content found for persona generation")
        return

    persona_name = (persona_name or "").strip() or "Speaker Name"

    # Sample many short chunks for tone
    sample_size = min(max(1, int(sample_count)), len(filtered))
    sampled = random.sample(filtered, sample_size)

    # Combine content
    sample_texts: list[str] = []
    for data in sampled:
        content = data.content[: max(1, int(max_chars_per_sample))]  # Limit each sample
        sample_texts.append(f"[{data.source_type.upper()}]\n{content}")

    combined_text = "\n\n---\n\n".join(sample_texts)
    bio = (bio_text or "").strip()

    # Generate persona prompt
    prompt = f"""You are writing a reusable system instruction for an AI assistant.

Persona name (explicit, do not change or guess): {persona_name}

BIO (high-signal grounding, use as-is; if empty, ignore):
{bio[:60000]}

SAMPLES:
{combined_text[:60000]}

Create a System Instruction that forces an AI to adopt this persona.
Start with exactly: "You are {persona_name}."

Describe:
- Their communication style
- Common phrases and expressions
- How they structure explanations
- Their approach to answering questions
- Emotional tone and energy level
- Key principles they emphasize

Rules:
- Return plain text only (no markdown, no headings, no bold/italics, no bullet symbols).
- End with a final line: END_SYSTEM_INSTRUCTION
"""

    logger.info(
        "Generating persona profile (samples=%d persona_name=%s)...", len(sampled), persona_name
    )
    persona_text = llm_generate(
        core_cfg=core_cfg,
        prompt=prompt,
        model=core_cfg.models.ingestion.persona.model,
        temperature=0.3,
        parse_json=False,
        max_tokens=max(512, int(max_output_tokens)),
    )

    if not isinstance(persona_text, str):
        persona_text = str(persona_text)

    if isinstance(persona_text, str) and "END_SYSTEM_INSTRUCTION" not in persona_text:
        logger.warning("Persona output may be truncated (missing END_SYSTEM_INSTRUCTION)")

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(persona_text, encoding="utf-8")
    logger.info("Saved persona profile to %s", output_path)
