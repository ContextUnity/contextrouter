"""Named Entity Recognition (NER) transformer.
Extracts named entities (persons, organizations, locations, dates, etc.) from text
and enriches document metadata with structured entity information.
Supports multiple backends:
- LLM-based extraction (high quality, uses existing model registry)
- Local models (spaCy, transformers) for fast, offline processing
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from typing import (
    ClassVar,
    NotRequired,
    Protocol,
    TypedDict,
    TypeGuard,
    override,
    runtime_checkable,
)

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.exceptions import ConfigurationError
from contextunity.core.parsing import json_loads
from contextunity.core.types import ContextUnitPayload, is_json_dict, is_object_list
from pydantic import BaseModel, ConfigDict

from contextunity.router.core import RouterConfig
from contextunity.router.core.registry import register_transformer
from contextunity.router.modules.models import model_registry
from contextunity.router.modules.models.types import ModelRequest, TextPart
from contextunity.router.modules.transformers._ingestion_helpers import payload_metadata

from .base import Transformer

logger = get_contextunit_logger(__name__)

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
    "MISC",  # Catch-all used by common NER models (e.g. CoNLL03)
}

_ENTITY_TYPE_ALIASES: dict[str, str] = {
    # CoNLL03-style
    "PER": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
    # spaCy common labels
    "PERSON": "PERSON",
    "GPE": "GPE",
    "NORP": "NORP",
    "FAC": "FAC",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "WORK_OF_ART",
    "LAW": "LAW",
    "LANGUAGE": "LANGUAGE",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
    "QUANTITY": "QUANTITY",
}


class NEREntity(TypedDict):
    """JSON-serializable NER entity record stored in metadata / struct_data."""

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    source: NotRequired[str]


@runtime_checkable
class _ObjectIterator(Protocol):
    def __next__(self) -> object: ...


class _SpacyDoc(Protocol):
    @property
    def ents(self) -> Sequence[object]: ...


class _SpacyLanguage(Protocol):
    def __call__(self, text: str) -> _SpacyDoc: ...


class _TransformersPipeline(Protocol):
    def __call__(self, text: str) -> object: ...


_SpacyLoad = Callable[[str], object]
_TransformersPipelineFactory = Callable[..., object]


def _is_spacy_load(value: object) -> TypeGuard[_SpacyLoad]:
    return callable(value)


def _is_transformers_factory(value: object) -> TypeGuard[_TransformersPipelineFactory]:
    return callable(value)


def _call_text_processor(value: object, text: str) -> object:
    if not callable(value):
        raise TypeError("Expected callable text processor")
    return value(text)


def _wrap_spacy_model(inner: object) -> _SpacyLanguage:
    class _DocAdapter:
        def __init__(self, inner_doc: object) -> None:
            self._inner: object = inner_doc

        @property
        def ents(self) -> Sequence[object]:
            raw = getattr(self._inner, "ents", ())
            if is_object_list(raw):
                return raw
            collected: list[object] = []
            iter_fn = getattr(raw, "__iter__", None)
            if callable(iter_fn):
                iterator_obj: object = iter_fn()
                if isinstance(iterator_obj, _ObjectIterator):
                    while True:
                        try:
                            collected.append(iterator_obj.__next__())
                        except StopIteration:
                            break
            return collected

    class _SpacyAdapter:
        def __call__(self, text: str) -> _DocAdapter:
            return _DocAdapter(_call_text_processor(inner, text))

    return _SpacyAdapter()


def _wrap_transformers_pipeline(inner: object) -> _TransformersPipeline:
    class _PipelineAdapter:
        def __call__(self, text: str) -> object:
            return _call_text_processor(inner, text)

    return _PipelineAdapter()


def _normalize_entity_type(raw: object) -> str:
    """Map a raw entity type label (CoNLL03 / spaCy style) to a canonical ``STANDARD_ENTITY_TYPES`` key."""
    t = str(raw or "").strip().upper()
    return _ENTITY_TYPE_ALIASES.get(t, t)


class NERConfig(BaseModel):
    """Configuration for NERTransformer."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    mode: str = "llm"
    model: str = ""
    entity_types: list[str] | None = None
    min_confidence: float = 0.5
    core_cfg: RouterConfig | None = None


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

    name: str = "ner"

    def __init__(self) -> None:
        """Initialize default ``NERConfig`` and lazy-load slots for spaCy / transformers models."""
        super().__init__()
        self.config: NERConfig = NERConfig()
        self._spacy_model: _SpacyLanguage | None = None
        self._transformers_pipeline: _TransformersPipeline | None = None

    @property
    def mode(self) -> str:
        """Active extraction backend: ``llm``, ``spacy``, or ``transformers``."""
        return self.config.mode

    @property
    def model(self) -> str:
        """Explicit model override (empty string = use default from registry)."""
        return self.config.model

    @property
    def entity_types(self) -> set[str] | None:
        """Normalized whitelist of entity types to keep, or ``None`` for all."""
        if not self.config.entity_types:
            return None
        parsed = {_normalize_entity_type(x) for x in self.config.entity_types if str(x).strip()}
        return parsed or None

    @property
    def min_confidence(self) -> float:
        """Confidence threshold below which local-model entities are dropped."""
        return self.config.min_confidence

    @property
    def _core_cfg(self) -> RouterConfig | None:
        """Optional ``RouterConfig`` override; resolved lazily if not set."""
        return self.config.core_cfg

    @_core_cfg.setter
    def _core_cfg(self, value: RouterConfig | None) -> None:
        """Store a ``RouterConfig`` override in the underlying ``NERConfig``."""
        self.config.core_cfg = value

    @override
    def configure(self, params: dict[str, object] | None) -> None:
        """Configure ner transformer."""
        super().configure(params)
        if params:
            # Pydantic handles type coercion safely
            self.config = NERConfig.model_validate(params)

    def _load_spacy_model(self) -> _SpacyLanguage:
        """Lazy-load a spaCy model: try Ukrainian (``uk_core_news_sm``) first, fall back to English."""
        if self._spacy_model is None:
            try:
                spacy = importlib.import_module("spacy")
                load_attr = getattr(spacy, "load", None)
                if not _is_spacy_load(load_attr):
                    raise ImportError("spaCy module does not expose callable load()")

                # Try to load Ukrainian model first, fallback to English
                try:
                    self._spacy_model = _wrap_spacy_model(load_attr("uk_core_news_sm"))
                    logger.info("Loaded spaCy Ukrainian model")
                except OSError:
                    try:
                        self._spacy_model = _wrap_spacy_model(load_attr("en_core_web_sm"))
                        logger.info("Loaded spaCy English model")
                    except OSError:
                        logger.warning(
                            "spaCy models not found. Install with: python -m spacy download en_core_web_sm"
                        )
                        raise
            except ImportError:
                logger.warning("spaCy not installed. Install with: uv add spacy")
                raise
        return self._spacy_model

    def _load_transformers_pipeline(self) -> _TransformersPipeline:
        """Lazy-load the ``xlm-roberta-large`` NER pipeline with simple aggregation."""
        if self._transformers_pipeline is None:
            try:
                transformers = importlib.import_module("transformers")
                pipeline_attr = getattr(transformers, "pipeline", None)
                if not _is_transformers_factory(pipeline_attr):
                    raise ImportError("transformers module does not expose callable pipeline()")

                # Use a multilingual model that supports Ukrainian
                self._transformers_pipeline = _wrap_transformers_pipeline(
                    pipeline_attr(
                        "ner",
                        model="xlm-roberta-large-finetuned-conll03-english",
                        aggregation_strategy="simple",
                    )
                )
                logger.info("Loaded transformers NER pipeline")
            except ImportError:
                logger.warning(
                    "transformers not installed. Install with: uv add transformers torch"
                )
                raise
        return self._transformers_pipeline

    def _extract_with_spacy(self, text: str) -> list[NEREntity]:
        """Extract entities using spacy."""
        try:
            nlp = self._load_spacy_model()
            doc = nlp(text)
            entities: list[NEREntity] = []

            for ent in doc.ents:
                entity_type = _normalize_entity_type(getattr(ent, "label_", ""))
                if self.entity_types and entity_type not in self.entity_types:
                    continue

                entities.append(
                    {
                        "text": str(getattr(ent, "text", "")),
                        "entity_type": entity_type,
                        "start": int(getattr(ent, "start_char", 0)),
                        "end": int(getattr(ent, "end_char", 0)),
                        "confidence": 1.0,  # spaCy doesn't provide confidence by default
                        "source": "spacy",
                    }
                )

            return entities
        except Exception as e:
            logger.error("spaCy NER extraction failed: %s", e)
            return []

    def _extract_with_transformers(self, text: str) -> list[NEREntity]:
        """Extract entities using transformers library."""
        try:
            pipe = self._load_transformers_pipeline()
            results = pipe(text)

            entities: list[NEREntity] = []
            if not is_object_list(results):
                return []
            for item in results:
                if not is_json_dict(item):
                    continue
                entity_type = _normalize_entity_type(
                    item.get("entity_group", item.get("label", "UNKNOWN"))
                )
                score_raw = item.get("score", 1.0)
                confidence = float(score_raw) if isinstance(score_raw, (int, float)) else 1.0

                if confidence < self.min_confidence:
                    continue
                if self.entity_types and entity_type not in self.entity_types:
                    continue

                start_raw = item.get("start", 0)
                end_raw = item.get("end", 0)
                entities.append(
                    {
                        "text": str(item.get("word", "")),
                        "entity_type": entity_type,
                        "start": int(start_raw) if isinstance(start_raw, (int, float)) else 0,
                        "end": int(end_raw) if isinstance(end_raw, (int, float)) else 0,
                        "confidence": confidence,
                        "source": "transformers",
                    }
                )

            return entities
        except Exception as e:
            logger.error("Transformers NER extraction failed: %s", e)
            return []

    async def _extract_with_llm(self, text: str) -> list[NEREntity]:
        """Prompt the default LLM to extract entities as a JSON array.

        Truncates input beyond 8 000 chars. Parses the response, validates each
        entity against ``STANDARD_ENTITY_TYPES`` and the configured whitelist,
        and returns only valid entries.
        """
        if not self._core_cfg:
            from contextunity.router.core import get_core_config

            self._core_cfg = get_core_config()

        cfg = self._core_cfg
        if not cfg:
            raise ConfigurationError("Failed to obtain core config.")

        # Truncate very long text
        if len(text) > 8000:
            text = text[:8000] + "\n\n[...truncated...]"

        prompt = f"""Extract all named entities from the following text. Return a JSON array of entities, each with:
- "text": the entity text
- "entity_type": one of {", ".join(sorted(STANDARD_ENTITY_TYPES))}
- "start": character position where entity starts
- "end": character position where entity ends

Text:
{text}

Return only valid JSON array, no markdown formatting."""

        try:
            model_key = self.model or cfg.models.default_llm
            llm = model_registry.get_llm_with_fallback(
                key=model_key,
                fallback_keys=[],
                strategy="fallback",
                config=cfg,
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

            parsed = json_loads(response_text)
            if not is_object_list(parsed):
                return []

            validated: list[NEREntity] = []
            for ent_obj in parsed:
                if not is_json_dict(ent_obj):
                    continue
                ent = ent_obj
                entity_type = _normalize_entity_type(ent.get("entity_type", ent.get("type", "")))
                if self.entity_types and entity_type not in self.entity_types:
                    continue
                if entity_type not in STANDARD_ENTITY_TYPES:
                    continue

                start_raw = ent.get("start", 0)
                end_raw = ent.get("end", 0)
                conf_raw = ent.get("confidence", 1.0)
                validated.append(
                    {
                        "text": str(ent.get("text", "")),
                        "entity_type": entity_type,
                        "start": int(start_raw) if isinstance(start_raw, (int, float)) else 0,
                        "end": int(end_raw) if isinstance(end_raw, (int, float)) else 0,
                        "confidence": float(conf_raw)
                        if isinstance(conf_raw, (int, float))
                        else 1.0,
                        "source": "llm",
                    }
                )

            return validated
        except Exception as e:
            logger.error("LLM NER extraction failed: %s", e)
            return []

    @override
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """Extract named entities from unit content and enrich metadata."""
        payload, metadata = payload_metadata(unit.payload)

        unit = self.with_provenance(unit, self.name)

        # Extract text from unit
        content = payload.get("content")
        text: str
        if is_json_dict(content):
            inner = content.get("content") or content.get("text") or ""
            text = str(inner)
        elif isinstance(content, str):
            text = content
        else:
            logger.warning("NER: unsupported content type %s", type(content))
            return unit

        if not text or len(text.strip()) < 10:
            logger.debug("NER: skipping short or empty content")
            return unit

        # Extract entities based on mode
        entities: list[NEREntity] = []
        if self.mode == "spacy":
            entities = self._extract_with_spacy(text)
        elif self.mode == "transformers":
            entities = self._extract_with_transformers(text)
        else:  # default to LLM
            entities = await self._extract_with_llm(text)

        if not entities:
            logger.debug("NER: no entities extracted")
            return unit

        # Group entities by type for easier access
        entities_by_type: dict[str, list[NEREntity]] = {}
        for ent in entities:
            entity_type = ent.get("entity_type", "UNKNOWN")
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(ent)

        # Store in metadata
        metadata_payload: ContextUnitPayload = dict(metadata)
        metadata_payload["ner_entities"] = [dict(ent) for ent in entities]
        metadata_payload["ner_entities_by_type"] = {
            k: [dict(ent) for ent in v] for k, v in entities_by_type.items()
        }
        metadata_payload["ner_entity_count"] = len(entities)
        metadata_payload["ner_mode"] = self.mode

        if "struct_data" in metadata_payload:
            struct_raw = metadata_payload["struct_data"]
            if is_json_dict(struct_raw):
                struct_data: ContextUnitPayload = dict(struct_raw)
                struct_data["ner_entities"] = [dict(ent) for ent in entities]
                metadata_payload["struct_data"] = struct_data

        payload["metadata"] = metadata_payload
        unit.payload = payload

        logger.debug(
            "NER: extracted %s entities (%s types) using %s",
            len(entities),
            len(entities_by_type),
            self.mode,
        )

        return unit


__all__ = ["NEREntity", "NERTransformer", "STANDARD_ENTITY_TYPES"]
