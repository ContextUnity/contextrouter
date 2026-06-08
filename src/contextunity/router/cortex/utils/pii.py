"""PII helpers — anonymize/deanonymize text around LLM calls.

Infrastructure layer: calls cortex.privacy.Anonymizer directly (in-process).
No LangChain tools, no gRPC — direct function calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from contextunity.core import get_contextunit_logger
from contextunity.core.types import is_object_dict, is_object_list
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = get_contextunit_logger(__name__)

__all__ = ["PiiSession", "should_apply_prompt_pii"]

if TYPE_CHECKING:
    from contextunity.router.cortex.privacy.anonymizer import Anonymizer
    from contextunity.router.modules.models.types import ModelRequest

_INFRA_KEYS: frozenset[str] = frozenset({"__token__", "metadata"})
_JSON_SCALAR_SAFE = (str, int, float, bool, type(None))


@runtime_checkable
class _KeyDestroyer(Protocol):
    """Encryption backend surface that supports key destruction."""

    def destroy_all_keys(self) -> None:
        """Destroy all ephemeral keys."""
        ...


def _user_roles(role: str) -> bool:
    """Check if the provided role corresponds to a user or human role.

    Args:
        role: The role identifier to verify.

    Returns:
        True if the role is a human/user role, False otherwise.
    """
    r = role.strip().lower()
    return r in ("user", "human")


def should_apply_prompt_pii() -> bool:
    """Return true when the active node token has permissions to use Router privacy masking.

    Checks the current access token scopes for both 'privacy:anonymize' and
    'privacy:deanonymize' permissions.

    Returns:
        True if the active token permits both anonymization and deanonymization, False otherwise.
    """
    try:
        from contextunity.router.core.context import get_current_access_token

        token = get_current_access_token()
    except Exception:
        return False
    return bool(
        token
        and token.has_permission("privacy:anonymize")
        and token.has_permission("privacy:deanonymize")
    )


class PiiSession:
    """Infrastructure-level PII anonymization session around LLM nodes.

    Calls cortex.privacy.Anonymizer directly, bypassing LangChain tool overhead.
    Instantiated per-node by secure_node.py when pii_masking is enabled in the configuration.
    PII operations are captured by BrainAutoTracer via callbacks.
    """

    session_id: str
    _anonymizer: Anonymizer
    _destroy_keys: _KeyDestroyer | None

    def __init__(self, session_id: str) -> None:
        """Store *session_id* and create an in-process ``Anonymizer`` with ephemeral AES-256 encryption."""
        self.session_id = session_id
        self._anonymizer, self._destroy_keys = self._create_anonymizer(session_id)

    @staticmethod
    def _pii_encryption_ttl_seconds() -> int:
        """Retrieve the configured TTL (Time-To-Live) for PII encryption keys.

        Returns:
            The TTL duration in seconds, defaulting to 60 if not configured.
        """
        try:
            from contextunity.router.core.config.main import get_core_config

            return int(get_core_config().privacy.pii_encryption_ttl_seconds)
        except Exception:
            return 60

    @staticmethod
    def _create_anonymizer(session_id: str) -> tuple[Anonymizer, _KeyDestroyer | None]:
        """Create an in-process Anonymizer with ephemeral AES-256 encryption.

        Args:
            session_id: The unique identifier for this anonymization context.

        Returns:
            An Anonymizer instance configured with a MappingStore and encryption backend.
        """
        from contextunity.router.cortex.privacy.anonymizer import Anonymizer
        from contextunity.router.cortex.privacy.masking import MappingStore
        from contextunity.router.cortex.privacy.masking.defaults import DEFAULT_MASKING_CONFIG
        from contextunity.router.cortex.privacy.masking.encryption import (
            EphemeralAES256Backend,
            PlaintextBackend,
        )

        ttl = PiiSession._pii_encryption_ttl_seconds()
        try:
            encryption = EphemeralAES256Backend(key_ttl_seconds=ttl)
        except ImportError:
            logger.warning(
                "cryptography not installed; PII values stored without encryption. Install `cryptography` for AES-256-GCM protection."
            )
            encryption = PlaintextBackend()

        store = MappingStore(
            session_id=session_id,
            db_path=":memory:",
            encryption=encryption,
        )
        destroy_keys = encryption if isinstance(encryption, _KeyDestroyer) else None
        return Anonymizer(store=store, config=DEFAULT_MASKING_CONFIG), destroy_keys

    def hide_messages(self, messages: list[object]) -> list[object]:
        """Anonymize a list of conversation messages by masking user string content.

        Args:
            messages: The list of raw message objects or dictionaries to process.

        Returns:
            A new list containing anonymized messages with PII values masked.
        """
        res: list[object] = []
        for item in messages:
            if isinstance(item, HumanMessage):
                text = self._anonymizer.anonymize(item.text).text
                res.append(HumanMessage(content=text))
            elif is_object_dict(item):
                role_value = item.get("role", "user")
                role = role_value if isinstance(role_value, str) else "user"
                content = item.get("content")
                if _user_roles(role) and isinstance(content, str):
                    masked = self._anonymizer.anonymize(content).text
                    new_item: dict[str, object] = dict(item)
                    new_item["content"] = masked
                    res.append(new_item)
                else:
                    res.append(item)
            else:
                res.append(item)
        return res

    def hide_dict(self, data: dict[str, object]) -> dict[str, object]:
        """Anonymize PII inside a dictionary, recursing into nested JSON structures.

        Args:
            data: The source dictionary to scan and mask.

        Returns:
            A new dictionary with PII values replaced by surrogate tokens, excluding
            metadata and authorization keys.
        """
        hidden: dict[str, object] = {}
        for key, value in data.items():
            hidden[key] = value if key in _INFRA_KEYS else self.hide_value(value)
        return hidden

    def hide_value(self, value: object) -> object:
        """Anonymize PII inside arbitrary JSON-like values (scalars, lists, dicts).

        Args:
            value: The data value to inspect and anonymize.

        Returns:
            The anonymized value.
        """
        if isinstance(value, str):
            return self.hide_text(value)
        if is_object_dict(value):
            hidden: dict[str, object] = {}
            for key, item in value.items():
                hidden[key] = item if key in _INFRA_KEYS else self.hide_value(item)
            return hidden
        if is_object_list(value):
            hidden_items: list[object] = []
            for item in value:
                hidden_items.append(self.hide_value(item))
            return hidden_items
        if isinstance(value, _JSON_SCALAR_SAFE):
            return value
        return value

    def hide_text(self, text: str) -> str:
        """Anonymize a single string by replacing PII with generated placeholders.

        Args:
            text: The raw input string to scan.

        Returns:
            The anonymized string with PII replaced by surrogate tokens.
        """
        return self._anonymizer.anonymize(text).text

    def hide_model_request(self, request: ModelRequest) -> ModelRequest:
        """Anonymize the ModelRequest payload before emitting telemetry or calling the provider.

        Args:
            request: The raw ModelRequest object to mask.

        Returns:
            A copy of the request with system prompts and text parts anonymized.
        """
        from contextunity.router.modules.models.types import TextPart

        update: dict[str, object] = {}
        if request.system:
            update["system"] = self.hide_text(request.system)

        parts: list[object] = []
        for part in request.parts:
            if isinstance(part, TextPart):
                parts.append(part.model_copy(update={"text": self.hide_text(part.text)}))
            else:
                parts.append(part)
        update["parts"] = parts
        return request.model_copy(update=update)

    def hide_trace_messages(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Anonymize a list of BaseMessage objects for tracing visibility.

        Args:
            messages: A list of messages to mask.

        Returns:
            A new list of messages containing masked text content.
        """
        return [self._hide_message(msg) for msg in messages]

    def emit_tool_result(
        self,
        tool_name: str,
        result: dict[str, object],
        *,
        node_name: str | None = None,
    ) -> None:
        """Emits prompt-level privacy operations as tool_result events in the Graph Journey.

        Args:
            tool_name: The name of the privacy tool/operation.
            result: The result payload to log.
            node_name: Optional name of the originating graph node.
        """
        try:
            from langchain_core.callbacks import dispatch_custom_event

            from contextunity.router.cortex.events import BrainEvent

            dispatch_custom_event(
                "brain_event",
                {
                    "event": BrainEvent(
                        type="tool_result",
                        node=node_name,
                        data={
                            "status": "ok",
                            "duration_ms": 0,
                            "tool_kind": "privacy",
                            "tool_binding": tool_name,
                            "handler": tool_name,
                            "source": "router_privacy",
                            "args": {"scope": "model_prompt"},
                            "result": result,
                            "error": None,
                        },
                    )
                },
            )
        except RuntimeError:
            pass

    def entities_masked_total(self) -> int:
        """Return the total number of unique PII entities masked in this session.

        Returns:
            The cumulative count of entities replaced.
        """
        return int(self._anonymizer.get_stats().get("entities_total", 0))

    def reveal_dict(self, data: dict[str, object]) -> dict[str, object]:
        """Deanonymize surrogate PII tokens inside a dictionary.

        Args:
            data: The dictionary containing masked/surrogate values.

        Returns:
            A new dictionary with PII placeholders restored to their original values.
        """
        out: dict[str, object] = {}
        for k, v in data.items():
            if k in _INFRA_KEYS:
                out[k] = v
            else:
                out[k] = self._reveal_value(v)
        return out

    def _reveal_value(self, value: object) -> object:
        """Recursively deanonymize surrogate PII tokens inside arbitrary JSON-like objects.

        Args:
            value: The data value to restore.

        Returns:
            The deanonymized value.
        """
        if isinstance(value, str):
            return self._anonymizer.deanonymize(value)
        if isinstance(value, BaseMessage):
            return self._reveal_message(value)
        if is_object_dict(value):
            revealed: dict[str, object] = {}
            for key, item in value.items():
                revealed[key] = self._reveal_value(item)
            return revealed
        if is_object_list(value):
            revealed_items: list[object] = []
            for item in value:
                revealed_items.append(self._reveal_value(item))
            return revealed_items
        return value

    def _reveal_message(self, msg: BaseMessage) -> BaseMessage:
        """Restore PII content inside a single BaseMessage.

        Args:
            msg: The message object to deanonymize.

        Returns:
            A copy of the message with the original content restored.
        """
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            return msg
        revealed = self._anonymizer.deanonymize(content)
        model_copy = getattr(msg, "model_copy", None)
        if callable(model_copy):
            new_msg = model_copy(update={"content": revealed})
            if isinstance(new_msg, BaseMessage):
                return new_msg
        # Fallback for non-Pydantic message implementations
        if isinstance(msg, HumanMessage):
            return HumanMessage(content=revealed)
        if isinstance(msg, AIMessage):
            return AIMessage(content=revealed)
        return msg

    def _hide_message(self, msg: BaseMessage) -> BaseMessage:
        """Mask PII content inside a single BaseMessage.

        Args:
            msg: The message object to anonymize.

        Returns:
            A copy of the message with masked text content.
        """
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            return msg
        masked = self.hide_text(content)
        model_copy = getattr(msg, "model_copy", None)
        if callable(model_copy):
            new_msg = model_copy(update={"content": masked})
            if isinstance(new_msg, BaseMessage):
                return new_msg
        if isinstance(msg, HumanMessage):
            return HumanMessage(content=masked)
        if isinstance(msg, AIMessage):
            return AIMessage(content=masked)
        return msg

    def reveal_text(self, text: str) -> str:
        """Deanonymize surrogate PII tokens inside a string.

        Args:
            text: The masked text string.

        Returns:
            The original text with PII values restored.
        """
        return self._anonymizer.deanonymize(text)

    def destroy(self) -> None:
        """Destroy the session, wiping PII mappings and ephemeral keys from RAM.

        After this call, no PII values from this session can be recovered.
        """
        self._anonymizer.reset()
        if self._destroy_keys is not None:
            self._destroy_keys.destroy_all_keys()
