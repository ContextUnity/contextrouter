"""Keyphrase extraction transformer.
Extracts keyphrases/keywords from text and enriches document metadata with
structured, JSON-serializable keyphrase information.
Design goals (project conventions):
- JSON-shaped outputs use TypedDict (no leaking Any).
- Outputs are StructData-safe (only primitives/lists/dicts).
- Provenance is recorded via ContextUnit provenance chain.
- Settings come from core config (no direct os.environ usage).
"""

from __future__ import annotations

from typing import ClassVar, NotRequired, TypedDict, override

from contextunity.core import ContextUnit, get_contextunit_logger
from contextunity.core.parsing import json_loads
from contextunity.core.types import ContextUnitPayload, is_json_dict, is_object_list
from pydantic import BaseModel, ConfigDict, Field

from contextunity.router.core import RouterConfig
from contextunity.router.core.registry import register_transformer
from contextunity.router.modules.models import model_registry
from contextunity.router.modules.models.types import ModelRequest, TextPart
from contextunity.router.modules.transformers._ingestion_helpers import payload_metadata

from .base import Transformer

logger = get_contextunit_logger(__name__)


class Keyphrase(TypedDict):
    """JSON-serializable keyphrase record stored in metadata / struct_data."""

    text: str
    score: float
    source: NotRequired[str]


def _normalize_phrase(s: object) -> str:
    """Collapse whitespace and strip leading/trailing punctuation from a raw keyphrase."""
    t = " ".join(str(s or "").strip().split())
    return t.strip(" \t\r\n,.;:!?'\"()[]{}")


class KeyphraseConfig(BaseModel):
    """Configuration for KeyphraseTransformer."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    mode: str = "llm"
    max_phrases: int = Field(default=15, ge=1, le=50)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    model: str = ""
    core_cfg: RouterConfig | None = None


@register_transformer("keyphrases")
class KeyphraseTransformer(Transformer):
    """Extract keyphrases from document content and enrich metadata.

    Configuration keys (all optional):
    - mode: currently only "llm" is supported (default)
    - max_phrases: maximum number of phrases to keep (default 15)
    - min_score: drop phrases with score below this (default 0.0)
    - core_cfg: provide Config override; otherwise uses get_core_config()
    """

    name: str = "keyphrases"

    def __init__(self) -> None:
        """Initialize default ``KeyphraseConfig`` — override via ``configure()``."""
        super().__init__()
        self.config: KeyphraseConfig = KeyphraseConfig()

    @property
    def mode(self) -> str:
        """Extraction backend (currently only ``llm`` is implemented)."""
        return self.config.mode

    @property
    def max_phrases(self) -> int:
        """Maximum number of keyphrases to retain (1–50)."""
        return self.config.max_phrases

    @property
    def min_score(self) -> float:
        """Score threshold below which extracted phrases are dropped."""
        return self.config.min_score

    @property
    def model(self) -> str:
        """Explicit model override (empty string = use default from registry)."""
        return self.config.model

    @property
    def _core_cfg(self) -> RouterConfig | None:
        """Optional ``RouterConfig`` override; resolved lazily if not set."""
        return self.config.core_cfg

    @_core_cfg.setter
    def _core_cfg(self, value: RouterConfig | None) -> None:
        """Store a ``RouterConfig`` override in the underlying ``KeyphraseConfig``."""
        self.config.core_cfg = value

    @override
    def configure(self, params: dict[str, object] | None) -> None:
        """Validate *params* via ``KeyphraseConfig`` and replace the active config."""
        super().configure(params)
        if not params:
            return

        self.config = KeyphraseConfig.model_validate(params)

    async def _extract_with_llm(self, text: str) -> list[Keyphrase]:
        """Prompt the default LLM to extract keyphrases as a JSON array.

        Truncates input beyond 8 000 chars. Deduplicates case-insensitively,
        filters by ``min_score`` and 80-char sanity bound, then sorts by
        descending score.
        """
        if not self._core_cfg:
            from contextunity.router.core import get_core_config

            self._core_cfg = get_core_config()

        cfg = self._core_cfg
        if not cfg:
            from contextunity.core.exceptions import ConfigurationError

            raise ConfigurationError("Failed to acquire core config")

        # Keep prompts bounded (cost + determinism)
        if len(text) > 8000:
            text = text[:8000] + "\n\n[...truncated...]"

        prompt = f"""Extract keyphrases from the text below.

Return ONLY a valid JSON array (no markdown), where each item is:
- "text": string keyphrase (2-6 words preferred; keep important proper nouns)
- "score": float in [0, 1] (higher = more important)

Rules:
- Return at most {self.max_phrases} items
- Prefer domain-specific terms over generic words
- Avoid duplicates (case-insensitive)

TEXT:
{text}
"""

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
            max_output_tokens=1024,
        )

        try:
            response = await llm.generate(request)
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

            data = json_loads(raw)
            if not is_object_list(data):
                return []

            out: list[Keyphrase] = []
            seen: set[str] = set()
            for item in data:
                if not is_json_dict(item):
                    continue
                phrase = _normalize_phrase(item.get("text"))
                if not phrase:
                    continue
                key = phrase.lower()
                if key in seen:
                    continue

                score_raw = item.get("score", 0.0)
                score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0

                if score < self.min_score:
                    continue
                score = max(0.0, min(score, 1.0))

                # Light sanity bounds (avoid dumping whole paragraphs)
                if len(phrase) > 80:
                    continue

                out.append({"text": phrase, "score": score, "source": "llm"})
                seen.add(key)

            out.sort(key=lambda x: (-x["score"], x["text"].lower()))
            return out[: self.max_phrases]
        except Exception:
            logger.exception("Keyphrase extraction failed")
            return []

    @override
    async def transform(self, unit: ContextUnit) -> ContextUnit:
        """Extract keyphrases from unit content via LLM and store them in
        ``metadata.keyphrases``, ``metadata.keyphrase_texts``, and (if present)
        ``struct_data``.
        """
        payload, metadata = payload_metadata(unit.payload)

        unit = self.with_provenance(unit, self.name)

        content = payload.get("content")
        if is_json_dict(content):
            primary = content.get("content")
            fallback = content.get("text")
            if isinstance(primary, str):
                text = primary
            elif isinstance(fallback, str):
                text = fallback
            else:
                text = ""
        elif isinstance(content, str):
            text = content
        else:
            logger.warning("keyphrases: unsupported content type %s", type(content))
            return unit

        if not text or len(text.strip()) < 20:
            return unit

        if self.mode != "llm":
            logger.warning("keyphrases: unsupported mode=%s; falling back to llm", self.mode)

        phrases = await self._extract_with_llm(text)
        if not phrases:
            return unit

        metadata_payload: ContextUnitPayload = dict(metadata)
        metadata_payload["keyphrases"] = [dict(p) for p in phrases]
        metadata_payload["keyphrase_texts"] = [p["text"] for p in phrases]
        metadata_payload["keyphrase_count"] = len(phrases)
        metadata_payload["keyphrase_mode"] = "llm"

        if "struct_data" in metadata_payload:
            struct_raw = metadata_payload["struct_data"]
            if is_json_dict(struct_raw):
                struct_data: ContextUnitPayload = dict(struct_raw)
                struct_data["keyphrases"] = [dict(p) for p in phrases]
                struct_data["keyphrase_texts"] = [p["text"] for p in phrases]
                struct_data["keyphrase_count"] = len(phrases)
                metadata_payload["struct_data"] = struct_data

        payload["metadata"] = metadata_payload
        unit.payload = payload
        return unit


__all__ = ["Keyphrase", "KeyphraseTransformer"]
