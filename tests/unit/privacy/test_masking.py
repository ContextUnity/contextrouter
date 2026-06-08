"""RED test: verify privacy masking module is importable from Router.

TDD Phase 1: This test MUST FAIL initially because cortex/privacy/ doesn't exist yet.
After we copy + adapt the masking engine, this becomes the smoke test.
"""

from __future__ import annotations

import pytest

from contextunity.router.core.exceptions import RouterPIIError


class TestMaskingCoreFunctionality:
    """Verify core masking behaviour works after migration."""

    def test_token_generator_random_hex(self) -> None:
        from contextunity.router.cortex.privacy.masking.tokens import TokenGenerator

        gen = TokenGenerator(style="random_hex")
        token = gen.generate("DOC")
        assert token.startswith("DOC_")
        assert len(token) == 8  # DOC_ + 4 hex chars

    def test_mapping_store_roundtrip(self) -> None:
        from contextunity.router.cortex.privacy.masking.store import MappingStore

        store = MappingStore(session_id="test-session", db_path=":memory:")
        token = store.get_or_create_token("John Doe", "doctor", "DOC")
        assert token.startswith("DOC_")

        # Same value → same token (consistency)
        token2 = store.get_or_create_token("John Doe", "doctor", "DOC")
        assert token == token2

        # Resolve back
        resolved = store.resolve_token(token)
        assert resolved == "John Doe"

        store.close()

    def test_session_stats_entity_counts(self) -> None:
        from contextunity.router.cortex.privacy.masking.store import MappingStore

        store = MappingStore(session_id="stats-session", db_path=":memory:")
        _ = store.get_or_create_token("Alice", "doctor", "DOC")
        _ = store.get_or_create_token("Bob", "doctor", "DOC")
        _ = store.get_or_create_token("x@y.z", "email", "EML")

        stats = store.get_session_stats()
        assert stats == {"doctor": 2, "email": 1}
        store.close()

    def test_masker_mask_text(self) -> None:
        from contextunity.router.cortex.privacy.masking.config import EntityRule, MaskingConfig
        from contextunity.router.cortex.privacy.masking.masker import PIIMasker
        from contextunity.router.cortex.privacy.masking.store import MappingStore

        config = MaskingConfig(
            text_entity_rules=[
                EntityRule(
                    entity_type="email",
                    prefix="EML",
                    pattern=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                ),
            ],
        )
        store = MappingStore(session_id="test", db_path=":memory:")
        masker = PIIMasker(store=store, config=config)

        masked = masker.mask_text("Contact john@example.com for details")
        assert "john@example.com" not in masked
        assert "EML_" in masked

        store.close()

    def test_unmasker_roundtrip(self) -> None:
        from contextunity.router.cortex.privacy.masking.config import EntityRule, MaskingConfig
        from contextunity.router.cortex.privacy.masking.masker import PIIMasker
        from contextunity.router.cortex.privacy.masking.store import MappingStore
        from contextunity.router.cortex.privacy.masking.unmasker import PIIUnmasker

        config = MaskingConfig(
            text_entity_rules=[
                EntityRule(
                    entity_type="phone",
                    prefix="PHN",
                    pattern=r"\+380\d{9}",
                ),
            ],
        )
        store = MappingStore(session_id="test", db_path=":memory:")
        masker = PIIMasker(store=store, config=config)
        unmasker = PIIUnmasker(store=store)

        original = "Call +380991234567 for info"
        masked = masker.mask_text(original)
        assert "+380991234567" not in masked

        unmasked = unmasker.unmask_text(masked)
        assert unmasked == original

        store.close()

    def test_encryption_ephemeral_roundtrip(self) -> None:
        from contextunity.router.cortex.privacy.masking.encryption import EphemeralAES256Backend

        backend = EphemeralAES256Backend(key_ttl_seconds=60)
        encrypted = backend.encrypt("sensitive data")
        decrypted = backend.decrypt(encrypted)
        assert decrypted == "sensitive data"

    def test_encryption_destroy_keys(self) -> None:
        from contextunity.router.cortex.privacy.masking.encryption import EphemeralAES256Backend

        backend = EphemeralAES256Backend(key_ttl_seconds=60)
        encrypted = backend.encrypt("sensitive data")
        backend.destroy_all_keys()
        assert backend.active_key_count == 0

        with pytest.raises(RouterPIIError, match="Unknown key_id"):
            backend.decrypt(encrypted)

    def test_scanner_detects_known_values(self) -> None:
        import hashlib

        from contextunity.router.cortex.privacy.masking.scanner import PostMaskScanner

        known = hashlib.sha256("John".encode()).hexdigest()
        scanner = PostMaskScanner(known_values_hashes={known})

        leaks = scanner.scan_text("The doctor John prescribed medicine")
        assert len(leaks) >= 1
        assert any(leak.leak_type == "known_value" for leak in leaks)

    def test_default_masking_config_loads(self) -> None:
        from contextunity.router.cortex.privacy.masking.defaults import DEFAULT_MASKING_CONFIG

        assert len(DEFAULT_MASKING_CONFIG.text_entity_rules) > 0
        # Should have at least person, phone, email rules
        entity_types = {r.entity_type for r in DEFAULT_MASKING_CONFIG.text_entity_rules}
        assert "person" in entity_types
        assert "phone" in entity_types
        assert "email" in entity_types
