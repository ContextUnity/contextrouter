"""Named Entity Recognition (NER) transformer.

Extracts named entities (persons, organizations, locations, dates, etc.) from text
and enriches document metadata with structured entity information.

Supports multiple backends:
- LLM-based extraction (high quality, uses existing model registry)
- Local models (spaCy, transformers) for fast, offline processing
"""

from __future__ import annotations

import json
import logging
from typing import Any

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.config import Config
from contextrouter.core.registry import register_transformer
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from .base import Transformer

logger = logging.getLogger(__name__)

# Standard NER entity types
STANDARD_ENTITY_TYPES = {
    "PERSON",  # People, characters
    "ORG",  # Organizations, companies
    "GPE",  # Geopolitical entities (countries, cities)
    "LOC",  # Locations (non-GPE)
    "DATE",  # Dates, times
    "MONEY",  # Monetary values
    "PERCENT",  # Percentages
    "QUANTITY",  # Measurements, quantities
    "EVENT",  # Events, occasions
    "PRODUCT",  # Products, brands
    "LAW",  # Legal documents, laws
    "LANGUAGE",  # Languages
    "WORK_OF_ART",  # Books, movies, art
    "FAC",  # Facilities, buildings
    "NORP",  # Nationalities, religious/political groups
}


@register_transformer("ner")
class NERTransformer(Transformer):
    """Extract named entities from document content and enrich metadata.

    Usage:
        transformer = NERTransformer()
        transformer.configure({
            "mode": "llm",  # or "spacy", "transformers"
            "entity_types": ["PERSON", "ORG", "LOC"],  # optional filter
            "min_confidence": 0.5,  # for local models
        })
        enriched_envelope = await transformer.transform(envelope)
    """

    name = "ner"

    def __init__(self) -> None:
        super().__init__()
        self.mode: str = "llm"
        self.entity_types: set[str] | None = None
        self.min_confidence: float = 0.5
        self._core_cfg: Config | None = None
        self._spacy_model: Any = None
        self._transformers_pipeline: Any = None

    def configure(self, params: dict[str, Any] | None) -> None:
        """Configure NER transformer."""
        super().configure(params)
        if params:
            self.mode = params.get("mode", "llm")
            entity_types = params.get("entity_types")
            if entity_types:
                self.entity_types = set(entity_types) if isinstance(entity_types, list) else None
            self.min_confidence = float(params.get("min_confidence", 0.5))
            cfg = params.get("core_cfg")
            if isinstance(cfg, Config):
                self._core_cfg = cfg

    def _load_spacy_model(self) -> Any:
        """Lazy-load spaCy model."""
        if self._spacy_model is None:
            try:
                import spacy

                # Try to load Ukrainian model first, fallback to English
                try:
                    self._spacy_model = spacy.load("uk_core_news_sm")
                    logger.info("Loaded spaCy Ukrainian model")
                except OSError:
                    try:
                        self._spacy_model = spacy.load("en_core_web_sm")
                        logger.info("Loaded spaCy English model")
                    except OSError:
                        logger.warning(
                            "spaCy models not found. Install with: python -m spacy download en_core_web_sm"
                        )
                        raise
            except ImportError:
                logger.warning("spaCy not installed. Install with: pip install spacy")
                raise
        return self._spacy_model

    def _load_transformers_pipeline(self) -> Any:
        """Lazy-load transformers pipeline."""
        if self._transformers_pipeline is None:
            try:
                from transformers import pipeline

                # Use a multilingual model that supports Ukrainian
                self._transformers_pipeline = pipeline(
                    "ner",
                    model="xlm-roberta-large-finetuned-conll03-english",
                    aggregation_strategy="simple",
                )
                logger.info("Loaded transformers NER pipeline")
            except ImportError:
                logger.warning("transformers not installed. Install with: pip install transformers")
                raise
        return self._transformers_pipeline

    def _extract_with_spacy(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using spaCy."""
        try:
            nlp = self._load_spacy_model()
            doc = nlp(text)
            entities = []

            for ent in doc.ents:
                entity_type = ent.label_
                if self.entity_types and entity_type not in self.entity_types:
                    continue

                entities.append(
                    {
                        "text": ent.text,
                        "type": entity_type,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 1.0,  # spaCy doesn't provide confidence by default
                    }
                )

            return entities
        except Exception as e:
            logger.error(f"spaCy NER extraction failed: {e}")
            return []

    def _extract_with_transformers(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using transformers library."""
        try:
            pipe = self._load_transformers_pipeline()
            results = pipe(text)

            entities = []
            for item in results:
                entity_type = item.get("entity_group", item.get("label", "UNKNOWN"))
                confidence = item.get("score", 1.0)

                if confidence < self.min_confidence:
                    continue
                if self.entity_types and entity_type not in self.entity_types:
                    continue

                entities.append(
                    {
                        "text": item.get("word", ""),
                        "type": entity_type,
                        "start": item.get("start", 0),
                        "end": item.get("end", 0),
                        "confidence": confidence,
                    }
                )

            return entities
        except Exception as e:
            logger.error(f"Transformers NER extraction failed: {e}")
            return []

    async def _extract_with_llm(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using LLM (via model registry)."""
        if not self._core_cfg:
            from contextrouter.core import get_core_config

            self._core_cfg = get_core_config()

        # Truncate very long text
        if len(text) > 8000:
            text = text[:8000] + "\n\n[...truncated...]"

        prompt = f"""Extract all named entities from the following text. Return a JSON array of entities, each with:
- "text": the entity text
- "type": one of {', '.join(sorted(STANDARD_ENTITY_TYPES))}
- "start": character position where entity starts
- "end": character position where entity ends

Text:
{text}

Return only valid JSON array, no markdown formatting."""

        try:
            model_key = self._core_cfg.models.default_llm
            llm = model_registry.get_llm_with_fallback(
                key=model_key,
                fallback_keys=[],
                strategy="fallback",
                config=self._core_cfg,
            )

            request = ModelRequest(
                parts=[TextPart(text=prompt)],
                temperature=0.0,
                max_output_tokens=2048,
            )

            response = await llm.generate(request)
            response_text = response.text.strip()

            # Remove markdown code fences if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

            entities = json.loads(response_text)
            if not isinstance(entities, list):
                return []

            # Validate and filter entities
            validated = []
            for ent in entities:
                if not isinstance(ent, dict):
                    continue
                entity_type = ent.get("type", "").upper()
                if self.entity_types and entity_type not in self.entity_types:
                    continue
                if entity_type not in STANDARD_ENTITY_TYPES:
                    continue

                validated.append(
                    {
                        "text": str(ent.get("text", "")),
                        "type": entity_type,
                        "start": int(ent.get("start", 0)),
                        "end": int(ent.get("end", 0)),
                        "confidence": float(ent.get("confidence", 1.0)),
                    }
                )

            return validated
        except Exception as e:
            logger.error(f"LLM NER extraction failed: {e}")
            return []

    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        """Extract named entities from envelope content and enrich metadata."""
        envelope = self._with_provenance(envelope, self.name)

        # Extract text from envelope
        content = envelope.content
        if isinstance(content, dict):
            text = content.get("content") or content.get("text") or ""
        elif isinstance(content, str):
            text = content
        else:
            logger.warning(f"NER: unsupported content type {type(content)}")
            return envelope

        if not text or len(text.strip()) < 10:
            logger.debug("NER: skipping short or empty content")
            return envelope

        # Extract entities based on mode
        entities: list[dict[str, Any]] = []
        if self.mode == "spacy":
            entities = self._extract_with_spacy(text)
        elif self.mode == "transformers":
            entities = self._extract_with_transformers(text)
        else:  # default to LLM
            entities = await self._extract_with_llm(text)

        if not entities:
            logger.debug("NER: no entities extracted")
            return envelope

        # Group entities by type for easier access
        entities_by_type: dict[str, list[dict[str, Any]]] = {}
        for ent in entities:
            entity_type = ent.get("type", "UNKNOWN")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(ent)

        # Store in metadata
        metadata = dict(envelope.metadata or {})
        metadata["ner_entities"] = entities
        metadata["ner_entities_by_type"] = entities_by_type
        metadata["ner_entity_count"] = len(entities)
        metadata["ner_mode"] = self.mode

        # Also add to struct_data if available (for ingestion pipeline)
        if "struct_data" in metadata:
            struct_data = dict(metadata["struct_data"])
            struct_data["ner_entities"] = entities
            metadata["struct_data"] = struct_data

        envelope.metadata = metadata

        logger.debug(
            f"NER: extracted {len(entities)} entities ({len(entities_by_type)} types) using {self.mode}"
        )

        return envelope


__all__ = ["NERTransformer", "STANDARD_ENTITY_TYPES"]
