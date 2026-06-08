"""Tests for privacy anonymizer and config (Phase 1 absorption)."""

from __future__ import annotations


class TestAnonymizer:
    """Verify Anonymizer works from new cortex.privacy location."""

    def test_anonymize_text_with_pii(self) -> None:
        from contextunity.router.cortex.privacy.anonymizer import AnonymizationResult, Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore, MaskingConfig
        from contextunity.router.cortex.privacy.masking.config import EntityRule

        config = MaskingConfig(
            text_entity_rules=[
                EntityRule(
                    entity_type="email",
                    prefix="EML",
                    pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                ),
            ],
        )
        store = MappingStore(session_id="test-anon", db_path=":memory:")
        anon = Anonymizer(store=store, config=config)

        result = anon.anonymize("Contact john@example.com for info")
        assert isinstance(result, AnonymizationResult)
        assert result.was_modified
        assert "john@example.com" not in result.text
        assert "EML_" in result.text
        assert result.entities_masked >= 1

        store.close()

    def test_anonymize_empty_text(self) -> None:
        from contextunity.router.cortex.privacy.anonymizer import Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore, MaskingConfig

        store = MappingStore(session_id="test-empty", db_path=":memory:")
        anon = Anonymizer(store=store, config=MaskingConfig())

        result = anon.anonymize("")
        assert result.text == ""
        assert not result.was_modified
        store.close()

    def test_deanonymize_restores_pii(self) -> None:
        from contextunity.router.cortex.privacy.anonymizer import Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore, MaskingConfig
        from contextunity.router.cortex.privacy.masking.config import EntityRule

        config = MaskingConfig(
            text_entity_rules=[
                EntityRule(entity_type="phone", prefix="PHN", pattern=r"\+380\d{9}"),
            ],
        )
        store = MappingStore(session_id="test-deanon", db_path=":memory:")
        anon = Anonymizer(store=store, config=config)

        result = anon.anonymize("Call +380991234567")
        restored = anon.deanonymize(result.text)
        assert restored == "Call +380991234567"

        store.close()

    def test_anonymize_dict(self) -> None:
        from contextunity.router.cortex.privacy.anonymizer import Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore, MaskingConfig
        from contextunity.router.cortex.privacy.masking.config import EntityRule

        config = MaskingConfig(
            column_rules={"name": EntityRule(entity_type="person", prefix="PER")},
        )
        store = MappingStore(session_id="test-dict", db_path=":memory:")
        anon = Anonymizer(store=store, config=config)

        result = anon.anonymize_dict({"name": "John Doe", "age": 30})
        assert result["name"].startswith("PER_")
        assert result["age"] == 30

        store.close()

    def test_stats_tracking(self) -> None:
        from contextunity.router.cortex.privacy.anonymizer import Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore, MaskingConfig
        from contextunity.router.cortex.privacy.masking.config import EntityRule

        config = MaskingConfig(
            text_entity_rules=[
                EntityRule(entity_type="email", prefix="EML", pattern=r"\w+@\w+\.\w+"),
            ],
        )
        store = MappingStore(session_id="test-stats", db_path=":memory:")
        anon = Anonymizer(store=store, config=config)

        anon.anonymize("test@example.com")
        anon.anonymize("other@example.com")
        stats = anon.get_stats()
        assert stats["anonymize_calls"] == 2
        assert stats["entities_total"] >= 2

        store.close()

    def test_reset_clears_mappings(self) -> None:
        from contextunity.router.cortex.privacy.anonymizer import Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore, MaskingConfig
        from contextunity.router.cortex.privacy.masking.config import EntityRule

        config = MaskingConfig(
            text_entity_rules=[
                EntityRule(entity_type="email", prefix="EML", pattern=r"\w+@\w+\.\w+"),
            ],
        )
        store = MappingStore(session_id="test-reset", db_path=":memory:")
        anon = Anonymizer(store=store, config=config)

        result = anon.anonymize("test@example.com")
        assert "EML_" in result.text

        anon.reset()
        stats = anon.get_stats()
        assert stats["anonymize_calls"] == 0
        assert stats["entities_total"] == 0

        store.close()
