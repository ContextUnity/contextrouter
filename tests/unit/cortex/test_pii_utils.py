"""Behavioral tests for cortex.utils.pii — PiiSession with direct Anonymizer calls.

Tests cover:
  - PiiSession: hide_text, reveal_text, hide_messages, hide_dict, reveal_dict
  - Infrastructure key exclusion (__token__, metadata, non-serializable objects)
  - Session lifecycle (destroy)
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from contextunity.router.cortex.utils.pii import PiiSession
from contextunity.router.modules.models.types import ModelRequest, TextPart

# ── PiiSession ────────────────────────────────────────────────────────────


class TestPiiSession:
    def test_session_creation(self):
        session = PiiSession(session_id="test-session")
        assert session.session_id == "test-session"
        assert session._anonymizer is not None

    def test_hide_text_passthrough(self):
        """Text without PII passes through unchanged."""
        session = PiiSession(session_id="s1")
        result = session.hide_text("hello world")
        assert result == "hello world"

    def test_reveal_text_passthrough(self):
        """Text without PII tokens passes through unchanged."""
        session = PiiSession(session_id="s1")
        result = session.reveal_text("hello world")
        assert result == "hello world"

    def test_hide_reveal_roundtrip(self):
        """Anonymize then deanonymize — original text restored."""
        session = PiiSession(session_id="roundtrip-1")
        hidden = session.hide_text("Contact john@example.com for info")
        assert "john@example.com" not in hidden
        revealed = session.reveal_text(hidden)
        assert "john@example.com" in revealed

    def test_hide_messages_masks_human_only(self):
        session = PiiSession(session_id="msg-1")

        msgs = [
            HumanMessage(content="my email is john@example.com"),
            AIMessage(content="assistant reply with john@example.com"),
        ]
        result = session.hide_messages(msgs)

        # Human message should be masked
        assert "john@example.com" not in result[0].content
        # AI message passes through unchanged
        assert result[1].content == "assistant reply with john@example.com"

    def test_hide_messages_dict_user_masks_email(self):
        session = PiiSession(session_id="dict-msg-1")
        msgs = [{"role": "user", "content": "reach me at john@example.com"}]
        result = session.hide_messages(msgs)
        assert isinstance(result[0], dict)
        assert "john@example.com" not in result[0]["content"]

    def test_hide_messages_dict_non_user_passthrough(self):
        session = PiiSession(session_id="dict-msg-2")
        msgs = [{"role": "assistant", "content": "john@example.com echoed"}]
        result = session.hide_messages(msgs)
        assert result[0]["content"] == "john@example.com echoed"

    def test_reveal_dict_preserves_message_types(self):
        """Deanonymize must not JSON-round-trip LangChain messages (type loss)."""
        session = PiiSession(session_id="msg-types-1")
        human = HumanMessage(content="john@example.com")
        ai = AIMessage(content="reply about john@example.com")
        hidden_msgs = session.hide_messages([human, ai])
        assert isinstance(hidden_msgs[0], HumanMessage)
        assert "john@example.com" not in (hidden_msgs[0].content or "")
        data = {"messages": hidden_msgs, "final_output": "ok"}
        revealed = session.reveal_dict(data)
        out_msgs = revealed["messages"]
        assert isinstance(out_msgs[0], HumanMessage)
        assert isinstance(out_msgs[1], AIMessage)
        assert "john@example.com" in (out_msgs[0].content or "")


class TestPiiDict:
    def test_hide_dict(self):
        session = PiiSession(session_id="dict-1")
        data = {"name": "John", "count": 42}
        result = session.hide_dict(data)
        assert "count" in result
        assert result["count"] == 42  # Non-PII preserved

    def test_reveal_dict_roundtrip(self):
        session = PiiSession(session_id="dict-2")
        original = {"greeting": "Hello john@example.com"}
        hidden = session.hide_dict(original)
        revealed = session.reveal_dict(hidden)
        assert "john@example.com" in revealed["greeting"]

    def test_hide_dict_masks_nested_json_payloads(self):
        session = PiiSession(session_id="nested-1")
        original = {
            "rows": [
                {
                    "doctor_name": "Іван Петренко",
                    "contact": "john@example.com",
                }
            ]
        }

        hidden = session.hide_dict(original)
        hidden_row = hidden["rows"][0]

        assert isinstance(hidden_row, dict)
        assert "Іван Петренко" not in str(hidden_row["doctor_name"])
        assert "john@example.com" not in str(hidden_row["contact"])
        revealed = session.reveal_dict(hidden)
        assert revealed == original

    def test_hide_model_request_masks_prompt_parts_and_system(self):
        session = PiiSession(session_id="request-1")
        request = ModelRequest(
            system="Ти допомагаєш лікарю Іван Петренко",
            parts=[TextPart(text="Contact john@example.com")],
        )

        hidden = session.hide_model_request(request)

        assert hidden is not request
        assert "Іван Петренко" not in (hidden.system or "")
        text_part = hidden.parts[0]
        assert isinstance(text_part, TextPart)
        assert "john@example.com" not in text_part.text
        response = session.reveal_text(f"{hidden.system}\n{text_part.text}")
        assert "Іван Петренко" in response
        assert "john@example.com" in response

    def test_infra_keys_excluded(self):
        """__token__ and metadata must be preserved unchanged."""
        session = PiiSession(session_id="infra-1")
        token_obj = object()  # Simulates a ContextToken
        data = {
            "__token__": token_obj,
            "metadata": {"session_id": "s1"},
            "text": "john@example.com",
        }
        result = session.hide_dict(data)
        assert result["__token__"] is token_obj
        assert result["metadata"] == {"session_id": "s1"}

    def test_non_serializable_values_excluded(self):
        """Non-JSON-serializable values (objects, sets) are preserved as-is."""
        session = PiiSession(session_id="non-serial-1")
        custom_obj = object()
        data = {"custom": custom_obj, "text": "hello"}
        result = session.hide_dict(data)
        assert result["custom"] is custom_obj
        assert result["text"] == "hello"

    def test_empty_dict_passthrough(self):
        session = PiiSession(session_id="empty-1")
        assert session.hide_dict({}) == {}
        assert session.reveal_dict({}) == {}

    def test_reveal_dict_infra_keys_excluded(self):
        """reveal_dict also preserves infrastructure keys."""
        session = PiiSession(session_id="reveal-infra-1")
        token_obj = object()
        data = {"__token__": token_obj, "text": "hello"}
        result = session.reveal_dict(data)
        assert result["__token__"] is token_obj


class TestPiiSessionLifecycle:
    def test_destroy_clears_mappings(self):
        session = PiiSession(session_id="lifecycle-1")
        # Create a mapping
        hidden = session.hide_text("john@example.com")
        assert "john@example.com" not in hidden
        # Destroy session
        session.destroy()
        # After destroy, deanonymize can't restore (mappings cleared)
        revealed = session.reveal_text(hidden)
        # The token will remain because mappings are gone
        assert hidden in revealed or "john@example.com" not in revealed


class TestPiiEncryptionFailClosed:
    def test_missing_cryptography_raises_without_dev_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from contextunity.core.exceptions import ConfigurationError

        def _raise_import(*_args: object, **_kwargs: object) -> None:
            raise ImportError("no cryptography")

        monkeypatch.setattr(
            "contextunity.router.cortex.privacy.masking.encryption.EphemeralAES256Backend",
            _raise_import,
        )
        monkeypatch.setattr(
            PiiSession,
            "_allow_plaintext_pii",
            staticmethod(lambda: False),
        )

        with pytest.raises(ConfigurationError, match="cryptography"):
            PiiSession(session_id="fail-closed")

    def test_allow_plaintext_pii_dev_flag_permits_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_import(*_args: object, **_kwargs: object) -> None:
            raise ImportError("no cryptography")

        monkeypatch.setattr(
            "contextunity.router.cortex.privacy.masking.encryption.EphemeralAES256Backend",
            _raise_import,
        )
        monkeypatch.setattr(
            PiiSession,
            "_allow_plaintext_pii",
            staticmethod(lambda: True),
        )

        session = PiiSession(session_id="dev-plain")
        assert session._anonymizer is not None
