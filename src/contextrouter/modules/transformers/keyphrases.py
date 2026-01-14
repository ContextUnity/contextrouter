"""Keyphrase extraction transformer.

Extracts keyphrases/keywords from text and enriches document metadata with
structured, JSON-serializable keyphrase information.

Design goals (project conventions):
- JSON-shaped outputs use TypedDict (no leaking Any).
- Outputs are StructData-safe (only primitives/lists/dicts).
- Provenance is recorded via Bisquit trace.
- Settings come from core config (no direct os.environ usage).
"""

from __future__ import annotations

import json
import logging
from typing import NotRequired, TypedDict

from contextrouter.core.bisquit import BisquitEnvelope
from contextrouter.core.config import Config
from contextrouter.core.registry import register_transformer
from contextrouter.modules.models import model_registry
from contextrouter.modules.models.types import ModelRequest, TextPart

from .base import Transformer

logger = logging.getLogger(__name__)


class Keyphrase(TypedDict):
    """JSON-serializable keyphrase record stored in envelope.metadata / struct_data."""

    text: str
    score: float
    source: NotRequired[str]


def _normalize_phrase(s: object) -> str:
    # Conservative normalization: keep spaces, remove leading/trailing punctuation.
    t = " ".join(str(s or "").strip().split())
    return t.strip(" \t\r\n,.;:!?'\"()[]{}")


@register_transformer("keyphrases")
class KeyphraseTransformer(Transformer):
    """Extract keyphrases from document content and enrich metadata.

    Configuration keys (all optional):
    - mode: currently only "llm" is supported (default)
    - max_phrases: maximum number of phrases to keep (default 15)
    - min_score: drop phrases with score below this (default 0.0)
    - core_cfg: provide Config override; otherwise uses get_core_config()
    """

    name = "keyphrases"

    def __init__(self) -> None:
        super().__init__()
        self.mode: str = "llm"
        self.max_phrases: int = 15
        self.min_score: float = 0.0
        self._core_cfg: Config | None = None

    def configure(self, params: dict[str, object] | None) -> None:
        super().configure(params)
        if not params:
            return

        self.mode = str(params.get("mode", "llm") or "llm").strip() or "llm"
        try:
            self.max_phrases = int(params.get("max_phrases", 15) or 15)
        except Exception:
            self.max_phrases = 15
        self.max_phrases = max(1, min(self.max_phrases, 50))

        try:
            self.min_score = float(params.get("min_score", 0.0) or 0.0)
        except Exception:
            self.min_score = 0.0
        self.min_score = max(0.0, min(self.min_score, 1.0))

        cfg = params.get("core_cfg")
        self._core_cfg = cfg if isinstance(cfg, Config) else None

    async def _extract_with_llm(self, text: str) -> list[Keyphrase]:
        if not self._core_cfg:
            from contextrouter.core import get_core_config

            self._core_cfg = get_core_config()

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
            max_output_tokens=1024,
        )

        try:
            response = await llm.generate(request)
            raw = (response.text or "").strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

            data = json.loads(raw)
            if not isinstance(data, list):
                return []

            out: list[Keyphrase] = []
            seen: set[str] = set()
            for item in data:
                if not isinstance(item, dict):
                    continue
                phrase = _normalize_phrase(item.get("text"))
                if not phrase:
                    continue
                key = phrase.lower()
                if key in seen:
                    continue

                try:
                    score = float(item.get("score", 0.0))
                except Exception:
                    score = 0.0

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

    async def transform(self, envelope: BisquitEnvelope) -> BisquitEnvelope:
        envelope = self._with_provenance(envelope, self.name)

        content = envelope.content
        if isinstance(content, dict):
            text = content.get("content") or content.get("text") or ""
        elif isinstance(content, str):
            text = content
        else:
            logger.warning("keyphrases: unsupported content type %s", type(content))
            return envelope

        if not text or len(text.strip()) < 20:
            return envelope

        if self.mode != "llm":
            logger.warning("keyphrases: unsupported mode=%s; falling back to llm", self.mode)

        phrases = await self._extract_with_llm(text)
        if not phrases:
            return envelope

        metadata = dict(envelope.metadata or {})
        metadata["keyphrases"] = phrases
        metadata["keyphrase_texts"] = [p["text"] for p in phrases]
        metadata["keyphrase_count"] = len(phrases)
        metadata["keyphrase_mode"] = "llm"
        envelope.metadata = metadata

        if "struct_data" in metadata and isinstance(metadata["struct_data"], dict):
            struct_data = dict(metadata["struct_data"])
            struct_data["keyphrases"] = phrases
            struct_data["keyphrase_texts"] = [p["text"] for p in phrases]
            struct_data["keyphrase_count"] = len(phrases)
            metadata["struct_data"] = struct_data
            envelope.metadata = metadata

        return envelope


__all__ = ["Keyphrase", "KeyphraseTransformer"]
