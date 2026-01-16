"""Main QA transformer that orchestrates all QA processing components."""

from __future__ import annotations

import logging
from typing import Any

from contextrouter.core import Config
from contextrouter.core.types import StructData

from .analyzer import QuestionAnalyzer
from .speaker import SpeakerProcessor
from .taxonomy_mapper import TaxonomyMapper

logger = logging.getLogger(__name__)


class QATransformer:
    """Main QA transformer that orchestrates all QA processing."""

    def __init__(self, core_cfg: Config, taxonomy: StructData | None = None) -> None:
        self.core_cfg = core_cfg
        self.speaker_processor = SpeakerProcessor(core_cfg)
        self.question_analyzer = QuestionAnalyzer(core_cfg)
        self.taxonomy_mapper = TaxonomyMapper(taxonomy)

    def transform_content(
        self,
        content: str,
        taxonomy: StructData | None = None,
        *,
        session_title: str = "",
    ) -> list[dict[str, Any]]:
        """Transform raw QA content into structured Q&A pairs."""
        try:
            # Step 1: Split by speakers
            interactions = self.speaker_processor.split_by_speakers_llm(content)

            # Step 2: Merge short interruptions
            merged_interactions = self.speaker_processor.merge_interruptions(interactions)

            # Step 3: Analyze segments for Q&A pairs
            if merged_interactions:
                analyses = self.question_analyzer._analyze_segments_batch(merged_interactions)

                # Step 4: Create structured records
                records = []
                for i, analysis in analyses.items():
                    if i < len(merged_interactions):
                        interaction = merged_interactions[i]

                        # Map to taxonomy
                        taxonomy_categories = self.taxonomy_mapper.map_to_taxonomy(
                            analysis.get("question", "") + " " + analysis.get("answer", ""),
                            taxonomy,
                        )

                        record = {
                            # Expected by downstream struct_data builders (Vertex requires it).
                            "session_title": session_title,
                            "source_title": session_title,
                            "question": analysis.get("question", ""),
                            "answer": analysis.get("answer", ""),
                            "speaker": interaction.get("speaker", ""),
                            "taxonomy_categories": taxonomy_categories,
                            "source_type": "qa",
                        }
                        records.append(record)

                return records

        except Exception as e:
            logger.error("QA transformation failed: %s", e)
            return []

        return []

    def _combine_interactions(self, interactions: list[dict[str, str]]) -> str:
        """Combine interactions back into text format."""
        parts = []
        for interaction in interactions:
            speaker = interaction.get("speaker", "")
            text = interaction.get("text", "")
            if speaker and text:
                parts.append(f"[{speaker}]: {text}")

        return "\n".join(parts)

    def _build_input_text(
        self, shadow_records: list[Any], taxonomy: StructData | None = None
    ) -> str:
        """Build input text from shadow records for processing."""
        texts = []

        for record in shadow_records:
            if hasattr(record, "content"):
                content = record.content
            elif isinstance(record, dict) and "content" in record:
                content = record["content"]
            else:
                continue

            # Add taxonomy context if available
            if taxonomy:
                categories = self.taxonomy_mapper.map_to_taxonomy(content, taxonomy)
                if categories:
                    content = f"[Topics: {', '.join(categories)}] {content}"

            texts.append(content)

        return "\n\n".join(texts)
