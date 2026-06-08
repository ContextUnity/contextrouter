"""Tests for transformer base, metadata, and keyphrase utilities.

Pure-function tests (no LLM calls):
- base.py: Transformer._with_provenance
- metadata.py: MetadataTransformer
- keyphrases.py: _normalize_phrase, KeyphraseConfig
"""

from __future__ import annotations

import pytest
from contextunity.core import ContextUnit

# ═══════════════════════════════════════════════════════════════════════
# Transformer base — _with_provenance
# ═══════════════════════════════════════════════════════════════════════


class TestTransformerProvenance:
    """Transformer._with_provenance appends step to provenance list."""

    def _make_transformer(self):
        from contextunity.router.modules.transformers.base import Transformer

        class Stub(Transformer):
            async def transform(self, unit: ContextUnit) -> ContextUnit:
                return unit

        return Stub()

    def test_appends_step(self):
        t = self._make_transformer()
        unit = ContextUnit(payload={"data": 1})
        result = t._with_provenance(unit, "step_1")
        assert "step_1" in result.provenance

    def test_multiple_steps(self):
        t = self._make_transformer()
        unit = ContextUnit(payload={})
        t._with_provenance(unit, "a")
        t._with_provenance(unit, "b")
        assert unit.provenance == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════
# MetadataTransformer
# ═══════════════════════════════════════════════════════════════════════


class TestMetadataTransformer:
    """MetadataTransformer normalizes payload and appends provenance."""

    @pytest.mark.asyncio
    async def test_dict_payload_preserved(self):
        from contextunity.router.modules.transformers.metadata import MetadataTransformer

        t = MetadataTransformer()
        unit = ContextUnit(payload={"key": "value"})
        result = await t.transform(unit)
        assert result.payload == {"key": "value"}
        assert "metadata_mapper" in result.provenance

    @pytest.mark.asyncio
    async def test_empty_payload_stays_empty(self):
        from contextunity.router.modules.transformers.metadata import MetadataTransformer

        t = MetadataTransformer()
        unit = ContextUnit(payload={})
        result = await t.transform(unit)
        assert result.payload == {}
        assert "metadata_mapper" in result.provenance


# ═══════════════════════════════════════════════════════════════════════
# Keyphrase utilities — _normalize_phrase, KeyphraseConfig
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizePhrase:
    """_normalize_phrase cleans and normalizes keyphrase text."""

    def test_basic(self):
        from contextunity.router.modules.transformers.keyphrases import _normalize_phrase

        assert _normalize_phrase("  machine learning  ") == "machine learning"

    def test_strips_punctuation(self):
        from contextunity.router.modules.transformers.keyphrases import _normalize_phrase

        assert _normalize_phrase('"deep learning",') == "deep learning"

    def test_collapses_whitespace(self):
        from contextunity.router.modules.transformers.keyphrases import _normalize_phrase

        assert _normalize_phrase("neural   network   architecture") == "neural network architecture"


class TestKeyphraseTransformerConfigure:
    """KeyphraseTransformer.configure applies config overrides."""

    def test_configure_with_params(self):
        from contextunity.router.modules.transformers.keyphrases import KeyphraseTransformer

        t = KeyphraseTransformer()
        t.configure({"max_phrases": 5, "min_score": 0.3})
        assert t.max_phrases == 5
        assert t.min_score == 0.3

    def test_configure_with_none(self):
        from contextunity.router.modules.transformers.keyphrases import KeyphraseTransformer

        t = KeyphraseTransformer()
        t.configure(None)
        assert t.max_phrases == 15  # default

    @pytest.mark.asyncio
    async def test_short_content_skipped(self):
        """Content < 20 chars → transform is a no-op."""
        from contextunity.router.modules.transformers.keyphrases import KeyphraseTransformer

        t = KeyphraseTransformer()
        unit = ContextUnit(payload={"content": "short"})
        result = await t.transform(unit)
        payload = result.payload
        assert isinstance(payload, dict)
        metadata = payload.get("metadata", {})
        assert "keyphrases" not in metadata

    @pytest.mark.asyncio
    async def test_none_content_skipped(self):
        from contextunity.router.modules.transformers.keyphrases import KeyphraseTransformer

        t = KeyphraseTransformer()
        unit = ContextUnit(payload={"content": None})
        result = await t.transform(unit)
        assert "keyphrases" in result.provenance
